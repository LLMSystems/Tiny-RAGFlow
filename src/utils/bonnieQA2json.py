import pandas as pd 
import json

def main(input_file, output_file):
    """
    Convert a Bonnie QA CSV file to JSON format.

    Parameters:
    input_file (str): Path to the input Bonnie QA CSV file.
    output_file (str): Path to the output JSON file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    dataset = []
    for index, row in df.iterrows():
        res = {
            "id": int(index)+1,
            "text": row['測試案例'],
            "metadata":{
                "question": row['標準問題'],
                "answer": row['標準答案']
                }
            }
        dataset.append(res)
    
    # Write the JSON data to the output file with proper UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main("data/bonnieQA/邦尼知識庫202511_processed.csv", "data/bonnieQA/dataset.json")