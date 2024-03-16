import csv
import random

def number_to_bits(number, bit_length):
    bits = [0 for _ in range(bit_length)]
    curr_id = bit_length - 1
    while number != 0 and curr_id >= 0:
        if number % 2 == 1:
            bits[curr_id] = 1
        curr_id -= 1
        number //= 2
    return bits

def is_palindrome(number_in_bits):
    return 1 if number_in_bits == number_in_bits[::-1] else 0

def get_data(max_number, bit_length):
    data = []
    for number in range(max_number):
        curr_data = number_to_bits(number, bit_length)
        single_data = [1]
        single_data.extend(curr_data)
        palindrome = is_palindrome(curr_data)
        single_data.append(palindrome)
        if palindrome == 1:                     # to overcome the majority of non-palindromic strength
            for _ in range(30):
                data.append(single_data)
        data.append(single_data)
    random.shuffle(data)
    return data

def add_to_csv(csv_file, data):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)



if __name__ == '__main__':
    max_number = 1024
    bit_length = 10
    csv_file = 'Assignment-1\\data.csv'
    data = get_data(max_number, bit_length)
    add_to_csv(csv_file, data)