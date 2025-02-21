import matplotlib.pyplot as plt

with open('data/3/rl.txt', 'r') as file:
    data = [float(line.strip().replace(',', '.')) for line in file]

ten_percent = int(len(data) * 0.1)
first_10_percent = data[:ten_percent]

plt.figure(figsize=(20, 8))

plt.subplot(2, 1, 1)
plt.plot(data, color='b', linewidth=0.5)
plt.title('y(t) from \'rl.txt\'')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(first_10_percent, color='r', linewidth=0.5)
plt.title('y(t) первые 10% данных ({} точек)'.format(len(first_10_percent)))
plt.grid(True)

# Отображение графиков
plt.tight_layout()  # Чтобы графики не перекрывались
plt.show()
