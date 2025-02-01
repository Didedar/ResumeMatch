from flask import Flask, render_template, redirect, request, jsonify, session, flash
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename
import re
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import logging
from werkzeug.exceptions import RequestEntityTooLarge
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import secrets
import psutil
from typing import Optional, Tuple
import magic
import chardet
import unicodedata
import google.generativeai as genai
import json
from dotenv import load_dotenv
from typing import TypedDict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

process = psutil.Process(os.getpid())
print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")

class Keywords(TypedDict):
    keywords: list[str]
    areas: list[str]
    summary: str

class PDFHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mime = magic.Magic(mime=True)
        
    def validate_file(self, file) -> Optional[str]:
        if not file:
            return "No file provided"
        file.seek(0)
        file_bytes = file.read(2048)
        mime_type = self.mime.from_buffer(file_bytes)
        file.seek(0)
        allowed_mimes = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
        }
        if mime_type not in allowed_mimes:
            return f"Invalid file type. Detected: {mime_type}"
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > 16 * 1024 * 1024:
            return "File size exceeds 16MB limit"
        return None

    def extract_text_from_pdf(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                if reader.is_encrypted:
                    return None, "PDF file is encrypted. Please provide an unencrypted PDF."
                extracted_text = []
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            encoding = chardet.detect(page_text.encode())['encoding']
                            if encoding and encoding.lower() != 'utf-8':
                                page_text = page_text.encode(encoding).decode('utf-8', errors='ignore')
                            extracted_text.append(page_text)
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                        continue
                text = "\n".join(extracted_text)
                if not text.strip():
                    return None, "No readable text found in PDF. The file might be scanned or contain only images."
                text = " ".join(line.strip() for line in text.split("\n") if line.strip())
                if len(text) < 50:
                    return None, "Extracted text is too short to be a valid resume."
                return text, None
        except Exception as e:
            error_msg = str(e)
            if "file has not been decrypted" in error_msg.lower():
                return None, "PDF file is encrypted. Please provide an unencrypted PDF."
            elif "pdf header not found" in error_msg.lower():
                return None, "Invalid or corrupted PDF file."
            else:
                self.logger.error(f"Error processing PDF: {error_msg}")
                return None, f"Error processing PDF: {error_msg}"

app = Flask(__name__)

secret_key = secrets.token_hex(32)

app.config['SECRET_KEY'] = secret_key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

uploads_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.config["UPLOADS_FOLDER"] = uploads_dir

logging.basicConfig(level=logging.DEBUG) # <--- УСТАНОВИТЕ DEBUG
logger = logging.getLogger(__name__)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return jsonify({"error": "File size exceeds the limit (16MB)"}), 413

@app.route("/")
def index():
    return render_template("index.html")
    
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    
    text = text.translate(str.maketrans("", "", ''.join(chr(i) for i in range(32))))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]", "", text)
    text = text.lower()

    return text


def detect_profession(text):
    """
    Определяет профессию на основе текста резюме с помощью Gemini API.
    """
    try:
        api_key = "AIzaSyDxjGIlo8DqwOcTyuLGj9APZxLpjG5Ne58"
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        Analyze the following resume text and determine the most likely profession/job role.
        Return the result as a JSON object with profession name in Russian and English.

        Resume text:
        {text}

        Return only a valid JSON object in the format:
        {{
            "profession_ru": "название профессии",
            "profession_en": "profession name"
        }}

        No explanations, only pure JSON.
        """

        response = model.generate_content(prompt)
        try:
            result = json.loads(response.text.replace('```json', '').replace('```', ''))
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini API response as JSON")
            logger.debug(f"Raw response: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error in Gemini profession detection: {str(e)}")
        return None
        
            
    except Exception as e:
        logger.error(f"Error in Gemini API processing: {str(e)}")
        return []

def compare_texts_with_gemini(resume_text, vacancy_texts):
    """
    Сравнивает текст резюме с ПАКЕТОМ текстов вакансий используя Gemini API за один запрос.
    """
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return {}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Объединяем тексты вакансий для отправки в одном запросе
        vacancies_text_block = "\n\n".join([
            f"Вакансия {i+1}:\n{vacancy_text}"
            for i, vacancy_text in enumerate(vacancy_texts)
        ])

        prompt = f"""
    Compare the following resume with MULTIPLE job vacancies and provide only a similarity score for EACH vacancy.
    Consider the following aspects for each vacancy to determine the similarity score:
    - Required skills match
    - Experience level match
    - Job responsibilities match
    - Overall compatibility

    Resume:
    {resume_text}

    Job Vacancies:
    {vacancies_text_block}

    **IMPORTANT INSTRUCTIONS:**
    You **MUST** respond with **valid JSON ONLY**.
    Do **NOT** include any text, explanations, or any characters outside of the JSON structure.
    The JSON response **MUST** be in the following strict format:

    ```json
    {{
        "Vacancy 1": {{
            "similarity_score": <score between 0 and 1>
        }},
        "Vacancy 2": {{
            "similarity_score": <score between 0 and 1>
        }},
        "Vacancy 3": {{
            "similarity_score": <score between 0 and 1>
        }},
        "Vacancy 4": {{
            "similarity_score": <score between 0 and 1>
        }},
        "Vacancy 5": {{
            "similarity_score": <score between 0 and 1>
        }}
    }}
    ```

    **Example of a valid JSON response:**
    ```json
    {{
        "Vacancy 1": {{
            "similarity_score": 0.85
        }},
        "Vacancy 2": {{
            "similarity_score": 0.60
        }}
    }}
    ```

    Return **ONLY** the valid JSON object as shown in the example above. **Do not add any extra text before or after the JSON.**
    """

        response = model.generate_content(prompt)
        similarity_scores = {}
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[len('```json'):]
            if response_text.endswith('```'):
                response_text = response_text[:-len('```')]
            if response_text.endswith('.'):
                response_text = response_text[:-1]

            result = json.loads(response_text)
            for i in range(1, len(vacancy_texts) + 1): # Итерируем по вакансиям
                vacancy_key = f"Vacancy {i}"
                if vacancy_key in result and 'similarity_score' in result[vacancy_key]: # Проверяем наличие ключей
                    similarity_scores[vacancy_key] = {'similarity_score': result[vacancy_key]['similarity_score']} # Сохраняем только score
                else:
                    logger.warning(f"В ответе от Gemini API для вакансии {i} не найден ключ 'similarity_score' или 'Vacancy {i}'. Установлено значение по умолчанию 0.0")
                    similarity_scores[vacancy_key] = {'similarity_score': 0.0} # Дефолтное значение score
        except json.JSONDecodeError as e:
            # ...
            # В случае ошибки парсинга, возвращаем пустые scores для всех вакансий
            for i in range(len(vacancy_texts)):
                similarity_scores[f"Vacancy {i+1}"] = {'similarity_score': 0.0} # <--- ИСПРАВЛЕНО: 'reason' удален

        # ...

        for i in range(1, len(vacancy_texts) + 1): # Итерируем по вакансиям
            vacancy_key = f"Vacancy {i}"
            if vacancy_key in result and 'similarity_score' in result[vacancy_key]: # Проверяем наличие ключей
                similarity_scores[vacancy_key] = {'similarity_score': result[vacancy_key]['similarity_score']} # Сохраняем только score
            else:
                logger.warning(f"В ответе от Gemini API для вакансии {i} не найден ключ 'similarity_score' или 'Vacancy {i}'. Установлено значение по умолчанию 0.0")
                similarity_scores[vacancy_key] = {'similarity_score': 0.0}

        return similarity_scores

    except Exception as e:
        logger.error(f"Error in Gemini API processing: {str(e)}")
        return {}

def search_vacancies(profession, top_n=100):
    """
    Поиск вакансий по определенной профессии
    """
    # Ensure top_n is an integer
    if isinstance(top_n, list):
        top_n = top_n[0] if top_n else 10
    
    try:
        per_page = int(top_n)
        if per_page <= 0:
            per_page = 10
    except (ValueError, TypeError):
        per_page = 10

    base_url = "https://api.hh.ru/vacancies"
    params = {
        "per_page": per_page,
        "area": 40,
        "text": profession.get('profession_ru', ''),  # Используем русское название профессии для поиска
        "search_field": "name"  # Ищем в названии вакансии
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        vacancies = data.get("items", [])
        if not vacancies:
            logger.warning("No vacancies found.")
            return []

        return [
            {
                "title": vacancy.get("name", "No title"),
                "url": vacancy.get("alternate_url", ""),
                "snippet": vacancy.get("snippet", {}).get("requirement", "No description"),
                "salary": vacancy.get("salary", {}),
                "employment_type": vacancy.get("employment", {}).get("name", "Не указано")
            }
            for vacancy in vacancies
        ]

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching vacancies: {str(e)}")
        return []

def compute_similarity_scores(resume_text, vacancies):
    """
    Вычисляет схожесть текста резюме с текстами вакансий через Gemini API.
    """
    try:
        if not resume_text or not vacancies:
            logger.error("Empty resume text or vacancies list")
            return {}

        vacancy_texts = []
        for vacancy in vacancies:
            title = (vacancy.get('title') or '').strip()
            snippet = (vacancy.get('snippet') or '').strip()
            combined_text = f"{title} {snippet}".strip()
            
            if combined_text: 
                vacancy_texts.append(combined_text)

        if not vacancy_texts:
            logger.error("No valid vacancy texts found")
            return {}

        similarity_scores = compare_texts_with_gemini(resume_text, vacancy_texts)

        return similarity_scores
            
    except Exception as e:
        logger.error(f"Unexpected error in compute_similarity_scores: {str(e)}")
        return {}

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        try:
            if "resume" not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files["resume"]
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            pdf_handler = PDFHandler()
            error = pdf_handler.validate_file(file)
            if error:
                return jsonify({"error": error}), 400

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOADS_FOLDER"], filename)

            try:
                file.save(file_path)

                # Извлекаем текст из файла
                if file.filename.endswith('.pdf'):
                    text, error = pdf_handler.extract_text_from_pdf(file_path)
                    if error:
                        return jsonify({"error": error}), 400
                elif file.filename.endswith(('.doc', '.docx')):
                    try:
                        doc = Document(file_path)
                        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                        if not text.strip():
                            return jsonify({"error": "No readable text found in document"}), 400
                    except Exception as e:
                        return jsonify({"error": f"Error processing document: {str(e)}"}), 400

                # Определяем профессию
                profession = detect_profession(text)
                if not profession:
                    return jsonify({"error": "Could not detect profession from resume"}), 400

                # Ищем вакансии по профессии
                vacancies = search_vacancies(profession)
                if not vacancies:
                    return jsonify({
                        "message": f"No vacancies found for profession: {profession.get('profession_ru')}",
                        "suggestions": [
                            "Try broadening your job search",
                            "Consider related positions",
                            "Check back later for new openings"
                        ]
                    }), 404
                
                limited_vacancies = vacancies[:30]
                vacancy_texts = [f"{v['title']} {v['snippet']}" for v in limited_vacancies]
                similarity_scores = compare_texts_with_gemini(text, vacancy_texts)
                
                # Добавляем scores к вакансиям и сортируем
                vacancy_list = []
                
                for i, vacancy in enumerate(limited_vacancies):
                    vacancy_with_score = vacancy.copy()
                    score_key = f"Vacancy {i+1}"
                    
                    if score_key in similarity_scores:
                        score_info = similarity_scores[score_key]
                        if isinstance(score_info, dict) and 'similarity_score' in score_info:
                            vacancy_with_score['similarity_score'] = float(score_info['similarity_score'])
                        else:
                            vacancy_with_score['similarity_score'] = 0.0
                    else:
                        vacancy_with_score['similarity_score'] = 0.0
                    
                    vacancy_list.append(vacancy_with_score)

                sorted_vacancies = sorted(
                    vacancy_list,
                    key=lambda x: x['similarity_score'],
                    reverse=True
                )

                # Возвращаем и профессию, и вакансии, и scores в одном объекте
                return jsonify({
                    "profession": profession,
                    "vacancies": sorted_vacancies,
                    "similarity_scores": similarity_scores
                })

            except Exception as e:
                logger.error(f"Unexpected error processing file: {str(e)}")
                return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
            finally:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error removing temporary file: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error in upload route: {str(e)}")
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
