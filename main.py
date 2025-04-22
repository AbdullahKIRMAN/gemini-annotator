from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from PIL import Image
import os
import json
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from typing import List, Dict, Tuple, Optional, Any
import time
import mimetypes
from ratelimiter import RateLimiter

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("annotation.log")
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_MAX_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 10
RETRY_WAIT_MAX = 40
API_RPM_LIMIT = 15 # <-- Dakikadaki istek limiti
API_REQUEST_PERIOD = 60 # saniye (1 dakika)

RETRYABLE_EXCEPTIONS = (
    genai_errors.ServerError,           # 5xx hataları
    genai_errors.ClientError,
    TimeoutError,                       # Genel zaman aşımı
    genai_errors.APIError,              # 503 Service Unavailable gibi diğer API hataları
)
PYTHON_MIN_VERSION = (3, 9)

# --- Helper Functions ---

def check_python_version():
    """Minimum Python sürümünü kontrol eder."""
    import sys
    if sys.version_info < PYTHON_MIN_VERSION:
        major, minor = PYTHON_MIN_VERSION
        logger.critical(
            f"Bu betik Python {major}.{minor} veya üzerini gerektirir. "
            f"Mevcut sürüm: {sys.version.split()[0]}"
        )
        sys.exit(1)

def get_mime_type(filename: str) -> str:
    """Dosya adından MIME türünü tahmin eder."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.webp':
        return 'image/webp'
    elif ext == '.bmp':
        return 'image/bmp'
    logger.warning(f"MIME türü tahmin edilemedi: {filename}. 'application/octet-stream' kullanılıyor.")
    return 'application/octet-stream'

def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Verilen görüntünün genişlik ve yüksekliğini döndürür."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            return img.width, img.height
    except FileNotFoundError:
        logger.error(f"Görüntü bulunamadı: {image_path}")
        return None
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
         logger.error(f"Görüntü dosyası okunamadı veya bozuk ({image_path}): {e}")
         return None
    except Exception as e:
        logger.error(f"Görüntü boyutları alınırken hata ({image_path}): {e}", exc_info=True)
        return None

def load_class_list(filepath: str) -> List[str]:
    """Verilen dosyadan sınıf listesini okur (her satır bir sınıf)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        if not classes:
            logger.error(f"Sınıf listesi dosyası ({filepath}) boş veya geçerli sınıf içermiyor.")
            raise ValueError("Sınıf listesi boş.")
        logger.info(f"{len(classes)} sınıf yüklendi: {', '.join(classes)}")
        return classes
    except FileNotFoundError:
        logger.error(f"Sınıf listesi dosyası bulunamadı: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Sınıf listesi dosyası okunurken hata ({filepath}): {e}", exc_info=True)
        raise

def create_class_mapping(class_list: List[str]) -> Dict[str, int]:
    """Sınıf listesinden sınıf adı -> sınıf ID eşlemesi oluşturur."""
    return {class_name.lower(): i for i, class_name in enumerate(class_list)}

def convert_gemini_to_yolo(
    gemini_annotations: List[Dict[str, Any]],
    class_mapping: Dict[str, int],
    image_width: int,
    image_height: int,
    image_path: str # Logging için
) -> List[str]:
    """
    Gemini API'sinden alınan etiketleri YOLO formatına dönüştürür.
    Gemini'nin [ymin, xmin, ymax, xmax] (0-1000 normalleştirilmiş) formatını bekler.
    """
    yolo_annotations = []
    if not isinstance(gemini_annotations, list):
         logger.warning(f"Beklenmeyen Gemini çıktı formatı (liste değil): {gemini_annotations} - Görüntü: {image_path}")
         return []

    for ann_index, annotation in enumerate(gemini_annotations):
        if not isinstance(annotation, dict):
            logger.warning(f"Beklenmeyen annotation formatı (sözlük değil): {annotation} at index {ann_index} - Görüntü: {image_path}")
            continue

        try:
            box_key = next((k for k in annotation if k.lower() == "box_2d"), None)
            label_key = next((k for k in annotation if k.lower() == "label"), None)

            if not box_key or not label_key:
                logger.warning(f"Eksik 'box_2d' veya 'label' anahtarı: {annotation} - Görüntü: {image_path}")
                continue

            box = annotation[box_key]
            label = str(annotation[label_key]).lower()

            if not isinstance(box, list) or len(box) != 4:
                logger.warning(f"Geçersiz 'box_2d' formatı: {box} - Görüntü: {image_path}")
                continue

            if label not in class_mapping:
                logger.debug(f"'{label}' sınıfı sınıf eşlemesinde bulunamadı. Atlama yapılıyor. Görüntü: {image_path}")
                continue

            class_id = class_mapping[label]

            try:
                ymin, xmin, ymax, xmax = map(float, box)
            except (ValueError, TypeError) as e:
                logger.warning(f"Geçersiz koordinat değeri: {box} ({e}) - Görüntü: {image_path}")
                continue

            if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmax <= 1000 and xmin < xmax and ymin < ymax):
                 logger.warning(f"Geçersiz veya sıra dışı koordinat değerleri (0-1000 aralığı dışında veya min>=max): {box} - Görüntü: {image_path}")
                 continue

            x_center = ((xmin + xmax) / 2) / 1000.0
            y_center = ((ymin + ymax) / 2) / 1000.0
            bbox_width = (xmax - xmin) / 1000.0
            bbox_height = (ymax - ymin) / 1000.0

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            bbox_width = max(0.0, min(1.0, bbox_width))
            bbox_height = max(0.0, min(1.0, bbox_height))

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            yolo_annotations.append(yolo_line)

        except Exception as e:
            logger.error(f"Etiket dönüştürme hatası ({annotation}): {e} - Görüntü: {image_path}", exc_info=True)
            continue

    return yolo_annotations

# --- Gemini API Interaction (Updated for google-genai >= 1.11.0 and Rate Limiting) ---

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=lambda retry_state: logger.warning(
        f"API hatası/limit ({type(retry_state.outcome.exception()).__name__}), yeniden deneniyor... " # <-- Log mesajı güncellendi
        f"(Deneme {retry_state.attempt_number}/{RETRY_ATTEMPTS})"
    )
)
def call_gemini_api(
    client: genai.Client,
    model_name: str,
    image_path: str,
    class_list: List[str],
    limiter: RateLimiter # <-- RateLimiter nesnesi eklendi
) -> Optional[List[Dict[str, Any]]]:
    """Gemini API'sini client.generate_content kullanarak çağırır ve rate limit uygular."""
    try:
        logger.debug(f"Gemini API çağrısı hazırlanıyor: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            if not image_data:
                logger.error(f"Görüntü dosyası boş: {image_path}")
                return None

        mime_type = get_mime_type(image_path)
        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)

        prompt = f"""
        Analyze the provided image and identify all objects belonging to the following classes: {', '.join(class_list)}.
        For each detected object, provide its bounding box and label.
        Return the results as a JSON list. Each item in the list should be a JSON object with two keys:
        1. "label": The name of the detected object class (must be one of the provided classes). Case-insensitive matching is acceptable for the label key itself (e.g., "Label" or "label").
        2. "box_2d": A list of four numbers representing the bounding box coordinates [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale. Case-insensitive matching is acceptable for the box key itself (e.g., "Box_2d" or "box_2d").

        Example for a single object:
        [
          {{"label": "kedi", "box_2d": [150, 200, 750, 800]}}
        ]
        If no objects from the list are found, return an empty JSON list: [].
        Provide ONLY the JSON list in your response, without any introductory text, explanations, or markdown formatting like ```json ... ```.
        """
        text_part = types.Part.from_text(text=prompt)
        contents = [image_part, text_part]

        generation_config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type='application/json',
        )

        # --- Rate Limiting ---
        logger.debug(f"Rate limiter bekleniyor (limit: {limiter.max_calls}/{limiter.period}s)... {image_path}")
        with limiter: # <-- Rate limiter burada devreye giriyor
            logger.debug(f"Rate limiter geçildi, API çağrılıyor... {image_path}")
            # API çağrısı
            response = client.models.generate_content(
                model=f'models/{model_name}',
                contents=contents,
                config=generation_config
                # request_options={"timeout": 60} # İsteğe bağlı zaman aşımı
            )
            logger.debug(f"API çağrısı tamamlandı. {image_path}")
        # --- /Rate Limiting ---


        if not response.candidates:
             logger.warning(f"Gemini'den geçerli aday yanıt alınamadı. Görüntü: {image_path}. Sebep: {response.prompt_feedback}")
             return None

        candidate = response.candidates[0]

        if candidate.finish_reason != types.FinishReason.STOP:
             logger.warning(f"Gemini yanıtı beklenmedik şekilde sonlandı. Sebep: {candidate.finish_reason.name}. Görüntü: {image_path}")
             if candidate.finish_reason == types.FinishReason.SAFETY:
                  logger.error(f"Yanıt güvenlik nedeniyle engellendi. Görüntü: {image_path}. Güvenlik Derecelendirmeleri: {candidate.safety_ratings}")
             # ResourceExhausted (429) hatası da FINISH_REASON_OTHER olarak gelebilir, tenacity zaten handle edecek
             elif candidate.finish_reason == types.FinishReason.RECITATION:
                  logger.warning(f"Yanıt alıntı nedeniyle engellendi. Görüntü: {image_path}")
             elif candidate.finish_reason == types.FinishReason.OTHER:
                  logger.warning(f"Yanıt 'OTHER' nedeniyle sonlandı (API limiti veya başka bir sorun olabilir). Görüntü: {image_path}")
             return None # Hata olarak işaretle

        if not candidate.content or not candidate.content.parts:
             logger.warning(f"Gemini yanıtında içerik veya bölüm bulunamadı. Görüntü: {image_path}")
             return []

        raw_text = candidate.content.parts[0].text.strip()

        # ```json ... ``` temizleme
        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json"):].strip()
        elif raw_text.startswith("```"):
             raw_text = raw_text[len("```"):].strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-len("```")].strip()

        if not raw_text:
             logger.info(f"Gemini'den boş metin yanıtı alındı (temizleme sonrası). Görüntü: {image_path}")
             return []

        logger.debug(f"Gemini Raw Text Response ({image_path}): {raw_text}")

        try:
            annotations = json.loads(raw_text)
            if not isinstance(annotations, list):
                logger.warning(f"Gemini'den JSON listesi bekleniyordu ama farklı bir yapı alındı: {type(annotations)} - Görüntü: {image_path}")
                if isinstance(annotations, dict) and "label" in annotations and "box_2d" in annotations:
                     logger.info(f"Tek nesne JSON'u listeye dönüştürülüyor. - Görüntü: {image_path}")
                     return [annotations]
                return None
            return annotations
        except json.JSONDecodeError as e:
            logger.error(f"Gemini yanıtı JSON olarak ayrıştırılamadı. Yanıt: '{raw_text}'. Hata: {e} - Görüntü: {image_path}")
            return None

    except genai_errors.APIError as e:
        logger.error(f"Gemini API Hatası ({type(e).__name__}) ({image_path}): {e}", exc_info=False)
        raise e
    except FileNotFoundError:
        logger.error(f"API çağrısı için görüntü bulunamadı: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Gemini API çağrısı sırasında beklenmedik hata ({image_path}): {e}", exc_info=True)
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             return None
        raise e

# --- Main Processing Logic ---
def process_image(
    image_path: str,
    output_dir: str,
    class_list: List[str],
    class_mapping: Dict[str, int],
    client: genai.Client,
    model_name: str,
    limiter: RateLimiter # <-- RateLimiter nesnesi eklendi
) -> Tuple[str, bool, Optional[str]]:
    """Tek bir görüntüyü işler: API çağrısı (rate limitli), dönüştürme, kaydetme."""
    start_time = time.monotonic()
    base_filename = os.path.basename(image_path)
    logger.info(f"İşleniyor: {base_filename}")

    dimensions = get_image_dimensions(image_path)
    if dimensions is None:
        return image_path, False, "Görüntü boyutları alınamadı veya dosya bozuk."
    img_width, img_height = dimensions

    try:
        # Limiter'ı API çağrısına ilet
        gemini_annotations = call_gemini_api(client, model_name, image_path, class_list, limiter)
    except RetryError as e:
        logger.error(f"API çağrısı tüm yeniden denemelere rağmen başarısız oldu ({base_filename}): {e}")
        gemini_annotations = None
    except Exception as e:
         logger.error(f"API çağrısı sırasında işlenemeyen hata ({base_filename}): {e}", exc_info=True)
         gemini_annotations = None

    if gemini_annotations is None:
        return image_path, False, "Gemini API çağrısı başarısız oldu, geçersiz yanıt döndü veya engellendi."

    annotation_filename = os.path.splitext(base_filename)[0] + ".txt"
    annotation_path = os.path.join(output_dir, annotation_filename)

    if not gemini_annotations:
        logger.info(f"Görüntüde ({base_filename}) belirtilen sınıflardan nesne bulunamadı veya API boş döndü.")
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(annotation_path, "w") as f:
                pass
            logger.info(f"Boş etiket dosyası oluşturuldu: {annotation_filename}")
            processing_time = time.monotonic() - start_time
            return image_path, True, f"Nesne bulunamadı, boş dosya oluşturuldu ({processing_time:.2f}s)"
        except IOError as e:
            logger.error(f"Boş etiket dosyası yazılamadı ({annotation_path}): {e}")
            return image_path, False, "Boş etiket dosyası yazılamadı."

    yolo_lines = convert_gemini_to_yolo(gemini_annotations, class_mapping, img_width, img_height, image_path)

    if not yolo_lines:
        logger.warning(f"Geçerli YOLO etiketleri oluşturulamadı (dönüştürme sonrası). Görüntü: {base_filename}")
        # Boş dosya oluşturmak yine de mantıklı olabilir, böylece hangi dosyaların işlendiği belli olur
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(annotation_path, "w") as f:
                pass
            logger.info(f"Boş etiket dosyası oluşturuldu (dönüştürme sonrası nesne yok): {annotation_filename}")
            processing_time = time.monotonic() - start_time
            return image_path, True, f"Dönüştürme sonrası nesne yok, boş dosya ({processing_time:.2f}s)"
        except IOError as e:
            logger.error(f"Boş etiket dosyası yazılamadı ({annotation_path}): {e}")
            return image_path, False, "Boş etiket dosyası yazılamadı (dönüştürme sonrası)."


    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(annotation_path, "w", encoding='utf-8') as f:
            for line in yolo_lines:
                f.write(line + "\n")
        processing_time = time.monotonic() - start_time
        logger.info(f"Etiketlendi ({len(yolo_lines)} nesne): {annotation_filename} ({processing_time:.2f}s)")
        return image_path, True, f"{len(yolo_lines)} nesne etiketlendi ({processing_time:.2f}s)"

    except IOError as e:
        logger.error(f"Etiket dosyası yazılamadı ({annotation_path}): {e}")
        return image_path, False, "Etiket dosyası yazılamadı."
    except Exception as e:
        logger.error(f"Etiket dosyası kaydedilirken beklenmedik hata ({annotation_path}): {e}", exc_info=True)
        return image_path, False, "Etiket dosyası kaydedilirken hata oluştu."


def process_dataset(
    image_dir: str,
    output_dir: str,
    class_list_path: str,
    api_key: str,
    model_name: str,
    max_workers: int,
    rpm_limit: int = API_RPM_LIMIT # <-- RPM limiti parametre olarak eklendi
):
    """Veri setindeki tüm görüntüleri paralel olarak işler (rate limitli)."""
    logger.info(f"Veri seti işleme başlatıldı: {image_dir}")
    logger.info(f"Çıktı klasörü: {output_dir}")
    logger.info(f"Kullanılacak model: {model_name}")
    logger.info(f"Maksimum işçi sayısı: {max_workers}")
    logger.info(f"API İstek Limiti: {rpm_limit} RPM") # <-- Log eklendi

    if not os.path.isdir(image_dir):
        logger.error(f"Görüntü klasörü bulunamadı veya bir dizin değil: {image_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        class_list = load_class_list(class_list_path)
        class_mapping = create_class_mapping(class_list)
    except Exception:
        logger.error("Sınıf listesi yüklenemedi, işlem durduruluyor.")
        return

    try:
        client = genai.Client(api_key=api_key)
        logger.info("Gemini Client başarıyla oluşturuldu.")
    except Exception as e:
        logger.error(f"Gemini Client oluşturulamadı: {e}", exc_info=True)
        return

    # --- Rate Limiter Oluşturma ---
    # RateLimiter(max_calls, period) saniyede max_calls kadar izin verir.
    # Biz dakikada (60 saniye) rpm_limit kadar istiyoruz.
    limiter = RateLimiter(max_calls=rpm_limit, period=API_REQUEST_PERIOD)
    logger.info(f"Rate Limiter oluşturuldu: {rpm_limit} çağrı / {API_REQUEST_PERIOD} saniye")
    # --- /Rate Limiter Oluşturma ---

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
    ]

    if not image_files:
        logger.warning(f"Görüntü klasöründe ({image_dir}) desteklenen formatta hiç görüntü bulunamadı.")
        return

    logger.info(f"Toplam {len(image_files)} görüntü işlenecek.")

    processed_count = 0
    success_count = 0
    failed_count = 0
    start_time_total = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Annotator') as executor:
        # Future'ları gönderirken limiter'ı da ilet
        futures = {
            executor.submit(process_image, img_path, output_dir, class_list, class_mapping, client, model_name, limiter): img_path
            for img_path in image_files
        }

        for future in as_completed(futures):
            img_path = futures[future]
            base_filename = os.path.basename(img_path)
            processed_count += 1
            try:
                _, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Başarısız: {base_filename} - Sebep: {message}")

                if processed_count % 20 == 0 or processed_count == len(image_files):
                     elapsed_time = time.monotonic() - start_time_total
                     rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                     logger.info(
                         f"İlerleme: {processed_count}/{len(image_files)} [{success_count}✓, {failed_count}✗] "
                         f"({elapsed_time:.1f}s, {rate:.2f} img/s)"
                     )

            except Exception as exc:
                failed_count += 1
                logger.error(f"Görüntü işlenirken ({base_filename}) beklenmedik görev hatası: {exc}", exc_info=True)

    end_time_total = time.monotonic()
    total_duration = end_time_total - start_time_total
    logger.info("-" * 40)
    logger.info("İşlem Tamamlandı!")
    logger.info(f"Toplam Görüntü: {len(image_files)}")
    logger.info(f"Başarılı: {success_count}")
    logger.info(f"Başarısız: {failed_count}")
    logger.info(f"Toplam Süre: {total_duration:.2f} saniye")
    if processed_count > 0 : logger.info(f"Ortalama Süre/Görüntü: {total_duration / processed_count:.2f} saniye")
    logger.info("-" * 40)

# --- Argüman Ayrıştırıcı ve Ana Çalıştırma Bloğu ---
def main():
    check_python_version()

    parser = argparse.ArgumentParser(
        description=f"Gemini API kullanarak bir görüntü veri setini YOLO formatında etiketler (google-genai >= 1.11.0, rate limitli). Python >= {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]} gereklidir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_dir", help="Etiketlenecek görüntüleri içeren klasörün yolu.")
    parser.add_argument("output_dir", help="Oluşturulacak YOLO etiket (.txt) dosyalarının kaydedileceği klasör.")
    parser.add_argument("class_list_path", help="Her satırda bir sınıf adı içeren .txt dosyasının yolu.")
    parser.add_argument("--api_key", help="Google Gemini API anahtarı. Belirtilmezse GOOGLE_API_KEY ortam değişkeninden okunur.", default=None)
    parser.add_argument("--model_name", help="Kullanılacak Gemini modeli.", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--max_workers", type=int, help="Paralel işleme için kullanılacak maksimum işçi (thread) sayısı.", default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--rpm", type=int, help="Dakikadaki maksimum API isteği limiti.", default=API_RPM_LIMIT) # <-- RPM argümanı
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Ayarlanacak loglama seviyesi.")

    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.critical("Google Gemini API anahtarı ne argüman olarak (--api_key) ne de GOOGLE_API_KEY ortam değişkeninde belirtilmemiş. İşlem durduruluyor.")
        return 1

    # RPM limitini process_dataset'e ilet
    process_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        class_list_path=args.class_list_path,
        api_key=api_key,
        model_name=args.model_name,
        max_workers=args.max_workers,
        rpm_limit=args.rpm # <-- Argümanı fonksiyona geç
    )
    return 0

# --- Script Execution (Example) ---
# Aşağıdaki kısım doğrudan çalıştırma içindir, argümanları buradan alır.
# Komut satırından çalıştırmak için bu kısmı kaldırıp `if __name__ == "__main__": main()` kullanın.

# Örnek Kullanım (Doğrudan çalıştırma için - Komut satırı yerine)
API_KEY='' # <-- BURAYA KENDİ API ANAHTARINIZI GİRİN
IMAGE_DIR='images' # <-- Görüntü klasörünüz
OUTPUT_DIR='output' # <-- Çıktı klasörünüz
CLASS_LIST_PATH='class_list.txt' # <-- Sınıf listeniz
MODEL_NAME='gemini-2.0-flash' # <-- Model adı (1.5-flash veya 2.0-flash gibi)
MAX_WORKERS=4 # <-- İşçi sayısı (RPM limitine dikkat!)
RPM_LIMIT_VALUE = 15 # <-- Dakikadaki istek limiti (Free Tier için 15)

if __name__ == "__main__":
    # API Anahtarını kontrol et
    api_key_to_use = API_KEY
    if not api_key_to_use or api_key_to_use == 'YOUR_API_KEY_HERE':
         env_key = os.environ.get("GOOGLE_API_KEY")
         if env_key:
              api_key_to_use = env_key
              logger.info("GOOGLE_API_KEY ortam değişkeninden API anahtarı kullanılıyor.")
         else:
              logger.critical("Lütfen script içindeki API_KEY değişkenini veya GOOGLE_API_KEY ortam değişkenini ayarlayın.")
              exit(1) # Hata koduyla çık

    # Log seviyesini INFO olarak ayarla (veya DEBUG isterseniz)
    logger.setLevel(logging.INFO)

    # Python sürümünü kontrol et
    check_python_version()

    # Ana işlemi başlat
    process_dataset(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        class_list_path=CLASS_LIST_PATH,
        api_key=api_key_to_use,
        model_name=MODEL_NAME,
        max_workers=MAX_WORKERS,
        rpm_limit=RPM_LIMIT_VALUE # RPM limitini fonksiyona geç
    )