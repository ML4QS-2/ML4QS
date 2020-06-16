import csv
import datetime

labels_and_seconds = [['Pause', 24], ['Salsa_1', 16], ['Bachata_1', 16], ['Kizomba_1', 16], ['Pause', 8],
                      ['Salsa_2', 16], ['Bachata_2', 16], ['Kizomba_2', 16]]


def create_labels(input):
    start = int(round(datetime.datetime(2020, 6, 16, 19, 0, 0, 0).timestamp()))
    i = 0
    results = []
    results.append(['label', 'label_start', 'label_end'])
    for inp in input:
        i += 1
        end_time = datetime.datetime.fromtimestamp(start) + datetime.timedelta(0, inp[1])
        results.append([inp[0], start * 1000000000, int(round(end_time.timestamp())) * 1000000000])
        start = int(round(end_time.timestamp()))

    return results


res = create_labels(labels_and_seconds)
print(res)
with open("./datasets/dance/Labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(res)
