import csv

def get_data_from_csv(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)

    X = [list(map(int, row[:-1])) for row in data]
    Y = [int(row[-1]) for row in data]
    
    return X,Y