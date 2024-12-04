import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/harsh/arjun/fedSSL-research/log_files/visualization_data.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['accuracy'], label='accuracy column vs epoch', color='blue')
plt.plot(df['epoch'], df['similarity'], label='similarity vs epoch', color='orange')
plt.xlabel('epochs')
plt.ylabel('values')
plt.title('Graph of accuracy and similarity vs epochs')
plt.legend()
plt.grid(True)
plt.savefig('/home/harsh/arjun/fedSSL-research/log_files/combined_plot.png')
plt.close()