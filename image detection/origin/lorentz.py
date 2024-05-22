import numpy as np
from scipy.optimize import curve_fit

# Lorentzian fonksiyonu tanımı
def lorentzian(x, x0, gamma, a):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)

# Verileri dosyadan okuma
input_file_path = 'image detection/origin/output.txt'

# x ve y verilerini tutacak listeler
x_data = []
y_data = []

# Dosyayı aç ve satır satır oku
with open(input_file_path, 'r') as file:
    for line in file:
        # Her satırdaki verileri boşlukla ayır
        parts = line.strip().split(' ')
        x_str, y_str = parts[0], parts[1]
        
        # x ve y verilerini düzenle
        x_str = x_str.replace(',', '.')
        y_str = y_str.replace(',', '.')
        
        # Float'a çevir ve listeye ekle
        x_data.append(float(x_str))
        y_data.append(float(y_str))

# Listeleri numpy array'e çevir
x_data = np.array(x_data)
y_data = np.array(y_data)

# x verisini 1'den başlayarak oluştur
x_fit_data = np.arange(1, len(y_data) + 1)

# Başlangıç tahminleri (x0, gamma, a)
initial_guess = [x_fit_data[np.argmax(y_data)], 1, max(y_data)]

# Fit işlemi
popt, _ = curve_fit(lorentzian, x_fit_data, y_data, p0=initial_guess)

# Fit edilmiş parametreler
x0_fit, gamma_fit, a_fit = popt
print(f"Fitted parameters: x0 = {x0_fit}, gamma = {gamma_fit}, a = {a_fit}")

# Fit edilmiş verileri hesaplama
y_fit = lorentzian(x_fit_data, *popt)

# Fit edilmiş verileri normalize etme (0-1 arası)
y_fit_normalized = (y_fit - min(y_fit)) / (max(y_fit) - min(y_fit))

# Fit edilmiş ve normalize edilmiş verileri kaydetme
output_file_path = 'fit_edilmis_veriler.txt'
output_data = np.column_stack((x_data, y_fit_normalized))
np.savetxt(output_file_path, output_data, delimiter=',', header='x,y_fit_normalized', comments='')

print(f"Fit edilmiş ve normalize edilmiş veriler '{output_file_path}' dosyasına kaydedildi.")