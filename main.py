
from sklearn.linear_model import Perceptron
from PIL import Image
from os import listdir
from termcolor import colored
import shutil


model = Perceptron()

train_x = []
train_y = []

px_data = []
for image_name in listdir('dataset'):
    image = Image.open(f'dataset/{image_name}').resize((70, 70)).convert('1')
    for x in range(image.width):
        for y in range(image.height):
            px_data.append(image.getpixel((x, y)))


    train_x.append(px_data)
    px_data = []

    train_y.append(int(image_name[0]))


def train():
    print(colored('training .....', 'green'))
    model.fit(train_x, train_y)

train()

def predict(image_path):
    px_data = []
    image = Image.open(image_path).resize((70, 70)).convert('1')
    for x in range(image.width):
        for y in range(image.height):
            px_data.append(image.getpixel((x, y)))

    return model.predict([px_data])[0]




# to test the script
def test():
    images = listdir('test')
    TOTAL = len(images)
    correct = 0

    if TOTAL > 0:
        for image_name in images:
            result = str(predict(f'test/{image_name}'))
            if image_name[0] != result:
                shutil.move(f'test/{image_name}', f'dataset/{image_name}')
                print(colored(f'{image_name} is not {result} moved to dataset', 'red'))
            else:
                print(colored(f'good {image_name} is {result}', 'green'))
                correct += 1

        print("*" * 20)
        print(colored(f' {correct} correct image', 'green'))

        if correct < TOTAL:
            test()
    else:
        print(colored('No images to test', 'red'))

#test()

def main():
    while True:
        image_path = input('image path :')
        image_name = image_path.split('.')[0]
        result = predict(image_path)
        print('the number is : ', result)
        a = input('Was the answer correct? [y/n]')
        if a == 'n':
            shutil.move(image_path, f'dataset/{image_name}.jpg')
            print(colored(f'{image_name} moved to dataset', 'green'))


if __name__ == '__main__':
    main()
