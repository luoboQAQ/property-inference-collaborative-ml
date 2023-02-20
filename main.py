# -*- coding: utf-8 -*-
from distributed_sgd import train_lfw
from inference_attack import evaluate_lfw

ATTRS = {
    'gender': ['Female', 'Male'],
    'smile': ['Not Smiling', 'Smiling'],
    'race': ['Asian', 'White', 'Black'],
    'glasses': ['Eyeglasses', 'Sunglasses', 'No Eyewear'],
    'age': ['Baby', 'Child', 'Youth', 'Middle Aged', 'Senior'],
    'hair': ['Black Hair', 'Blond Hair', 'Brown Hair', 'Bald']}
INDEX_ATTRS = ['gender', 'smile', 'race', 'glasses', 'age', 'hair']


def main():
    print("Please choose the main task you want to perform:")
    print("1.gender\t2.smile\t3.race\t4.glasses\t5.age\t6.hair")
    task = input("Please input the number:")
    task = INDEX_ATTRS[int(task)-1]
    print("Please choose the attribute you want to perform:")
    print("1.gender\t2.smile\t3.race\t4.glasses\t5.age\t6.hair")
    attr = input("Please input the number:")
    attr = INDEX_ATTRS[int(attr)-1]
    print("Please choose the property you want to perform:")
    for i in range(len(ATTRS[attr])):
        print(str(i+1)+"."+ATTRS[attr][i])
    prop_id = input("Please input the number:")
    prop_id = int(prop_id)-1
    print("[INFO] Task: {}, Attribute: {}, Property: {}".format(task, attr, prop_id))
    # Train the model
    train_lfw(task, attr, prop_id)

    # Evaluate the model
    evaluate_lfw(task, attr, prop_id)


if __name__ == '__main__':
    main()
