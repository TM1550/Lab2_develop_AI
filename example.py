from transformers import pipeline

from functions_QA2 import smart_qa_with_aggregation, calculate_f1_for_json_dataset

#Пример использования
if __name__ == "__main__":
    # Инициализируем модель
    model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    context = "In school, children learn to read and write. Writing helps us to remember things and to communicate with other people. We can write stories, letters, or lists.\nTo make words on a page, you need a tool. You use a pencil or a pen to write on paper. A pencil uses graphite, and a pen uses ink.\nBefore there were pens and pencils, people used feathers or brushes. Today, many people also type on computers and phones instead of writing by hand"
    question = "What do you use to write on paper?"
    
    result = smart_qa_with_aggregation(model, question, context)
    
    print("Детали ответа:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Файл
    original_file = "dataset.json"

    # Оцениваем производительность на отфильтрованном датасете
    print("\nEvaluating on filtered dataset...")
    f1, accuracy, stats = calculate_f1_for_json_dataset(original_file, model, use_min_answer=True)

    print(f"\n=== Results ===")
    print(f"Filtered dataset - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")