import sys
from theme_16.core import create_file, create_folder, get_list, delete_file, copy_file, save_info, guess_the_number
import torch
import seaborn as sns
import matplotlib.pyplot as plt

command = sys.argv[1]

def handle_missing_argument():
    print('Не указан путь до файла или папки')

if command == 'list':
    get_list('theme_16')



  
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        y_pred = model(data).squeeze()
        loss = criterion(y_pred, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

   
    sns.set(style="whitegrid")
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels.numpy(), palette="Set2")
    plt.title("Scatter plot of the data with decision boundary")
    
  
    plt.savefig('data_analysis_plot.png')
    print('Анализ данных завершен. График сохранен в data_analysis_plot.png.')

else:
    print('Неизвестная команда. Введите help, чтобы увидеть доступные команды.')
