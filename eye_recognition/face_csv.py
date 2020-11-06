import csv


def write_csv(csv_file, id, name):
    with open(csv_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([id, name])

        

def get_csv_to_dic(csv_file, dic_name={}):
    with open(csv_file)as f:
        reader = csv.reader(f)
        for key, value in reader:
            dic_name[key] = value
    return dic_name



if __name__ == "__main__":
    csv_file = "test.csv"
    id = 3
    name = "tamura"
    dic_name = get_csv_to_dic(csv_file)
    print(dic_name)

    write_csv(csv_file , id, name)