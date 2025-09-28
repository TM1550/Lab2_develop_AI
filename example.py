from transformers import pipeline

from functions_QA2 import smart_qa_with_aggregation, calculate_f1_for_json_dataset, read_txt_file

#Пример использования
if __name__ == "__main__":
    # Инициализируем модель
    model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    document="document.txt"
    context = read_txt_file(document)
    question = "What do you use to write on paper?"
    
    result = smart_qa_with_aggregation(model, question, context)
    
    print("Детали ответа:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Файл
    original_file = "dataset.json"

    # Оцениваем производительность на отфильтрованном датасете
    print("\nEvaluating on dataset...")
    f1, accuracy, stats = calculate_f1_for_json_dataset(original_file, model, use_min_answer=True)

    print(f"\n=== Results ===")
    print(f"Dataset - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")