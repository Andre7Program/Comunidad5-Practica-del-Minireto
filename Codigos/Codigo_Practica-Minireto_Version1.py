import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy.ndimage as ndimage


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Minireto. Extracción de Componentes Conexas")
        self.root.geometry("1400x1000")
        self.root.configure(bg='#f0f0f0')

        # Variables para almacenar imágenes
        self.original_image = None
        self.processed_image = None
        self.image2 = None
        self.gray_image = None
        self.binary_image = None

        # Variables para histogramas
        self.histogram_figure = None
        self.histogram_canvas = None

        # Variables para mantener referencia a las imágenes tkinter
        self.original_photo = None
        self.processed_photo = None

        self.setup_ui()

    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configurar grid
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Título
        title_label = ttk.Label(main_frame, text="Transformaciones Lógicas y Etiquetado de Componentes Conexas",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Frame principal dividido en izquierda (controles) y derecha (visualización)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))


        main_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)


        controls_frame = ttk.LabelFrame(left_frame, text="Controles", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        controls_frame.columnconfigure(0, weight=1)

        # Botones de carga y eliminación de imágenes
        load_frame = ttk.Frame(controls_frame)
        load_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(load_frame, text="Cargar Imagen 1", command=self.load_image1).grid(row=0, column=0, padx=2)
        ttk.Button(load_frame, text="Cargar Imagen 2", command=self.load_image2).grid(row=0, column=1, padx=2)
        ttk.Button(load_frame, text="Eliminar Imagen 1", command=self.delete_image1).grid(row=1, column=0, padx=2,
                                                                                          pady=2)
        ttk.Button(load_frame, text="Eliminar Imagen 2", command=self.delete_image2).grid(row=1, column=1, padx=2,
                                                                                          pady=2)
        ttk.Button(load_frame, text="Eliminar Todas", command=self.delete_all_images).grid(row=1, column=2, padx=2,
                                                                                           pady=2)
        ttk.Button(load_frame, text="Reset Procesada", command=self.reset_image).grid(row=0, column=2, padx=2)

        # Operaciones aritméticas
        arith_frame = ttk.LabelFrame(controls_frame, text="Operaciones Aritméticas")
        arith_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(arith_frame, text="Suma", command=lambda: self.arithmetic_operation('add')).grid(row=0,
                                                                                                                  column=0,
                                                                                                                  padx=2,
                                                                                                                  pady=2)
        ttk.Button(arith_frame, text="Resta", command=lambda: self.arithmetic_operation('subtract')).grid(
            row=0, column=1, padx=2, pady=2)
        ttk.Button(arith_frame, text="Multiplicación",
                   command=lambda: self.arithmetic_operation('multiply')).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(arith_frame, text="División",
                   command=lambda: self.arithmetic_operation('divide')).grid(row=0, column=3, padx=2, pady=2)

        # Operaciones lógicas
        logic_frame = ttk.LabelFrame(controls_frame, text="Operaciones Lógicas")
        logic_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(logic_frame, text="AND", command=lambda: self.logical_operation('and')).grid(row=0, column=0, padx=2,
                                                                                                pady=2)
        ttk.Button(logic_frame, text="OR", command=lambda: self.logical_operation('or')).grid(row=0, column=1, padx=2,
                                                                                              pady=2)
        ttk.Button(logic_frame, text="XOR", command=lambda: self.logical_operation('xor')).grid(row=0, column=2, padx=2,
                                                                                                pady=2)
        ttk.Button(logic_frame, text="NOT", command=lambda: self.logical_operation('not')).grid(row=0, column=3, padx=2,
                                                                                                pady=2)


        preprocess_frame = ttk.LabelFrame(controls_frame, text="Preprocesamiento")
        preprocess_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(preprocess_frame, text="Escala de Grises", command=self.convert_to_grayscale).grid(row=0, column=0,
                                                                                                      padx=2, pady=2)

        # umbralización personalizada
        threshold_frame = ttk.Frame(preprocess_frame)
        threshold_frame.grid(row=0, column=1, columnspan=3, padx=5)

        ttk.Button(threshold_frame, text="Umbralizar", command=self.threshold_image).grid(row=0, column=0, padx=2)
        ttk.Button(threshold_frame, text="Umbral Personalizado", command=self.custom_threshold).grid(row=0, column=1,
                                                                                                     padx=2)

        # Etiquetado de componentes
        labeling_frame = ttk.LabelFrame(controls_frame, text="Etiquetado de Componentes Conexas")
        labeling_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(labeling_frame, text="Vecindad 4", command=lambda: self.connected_components(4)).grid(row=0,
                                                                                                         column=0,
                                                                                                         padx=2, pady=2)
        ttk.Button(labeling_frame, text="Vecindad 8", command=lambda: self.connected_components(8)).grid(row=0,
                                                                                                         column=1,
                                                                                                         padx=2, pady=2)
        ttk.Button(labeling_frame, text="Contornear Objetos", command=self.contour_objects).grid(row=0, column=2,
                                                                                                 padx=2, pady=2)

        # Histogramas
        histogram_frame = ttk.LabelFrame(controls_frame, text="Análisis de Histogramas")
        histogram_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(histogram_frame, text="Histograma RGB", command=lambda: self.show_histogram('rgb')).grid(row=0,
                                                                                                            column=0,
                                                                                                            padx=2,
                                                                                                            pady=2)
        ttk.Button(histogram_frame, text="Histograma Grises", command=lambda: self.show_histogram('gray')).grid(row=0,
                                                                                                                column=1,
                                                                                                                padx=2,
                                                                                                                pady=2)
        ttk.Button(histogram_frame, text="Histograma Binarizada", command=lambda: self.show_histogram('binary')).grid(
            row=0, column=2, padx=2, pady=2)
        ttk.Button(histogram_frame, text="Comparar Histogramas", command=self.compare_histograms).grid(row=0, column=3,
                                                                                                       padx=2, pady=2)
        ttk.Button(histogram_frame, text="Limpiar Histograma", command=self.clear_histogram).grid(row=1, column=0,
                                                                                                  columnspan=4, pady=5)

        info_frame = ttk.LabelFrame(left_frame, text="Información y Resultados", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

        self.info_text = tk.Text(info_frame, height=8, width=50, font=('Consolas', 9))
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)



        images_frame = ttk.LabelFrame(right_frame, text="Visualización de Imágenes", padding="10")
        images_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)


        original_frame = ttk.LabelFrame(images_frame, text="Imagen Original", padding="5")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)

        self.original_canvas = tk.Canvas(original_frame, bg='lightgray', width=400, height=300)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


        processed_frame = ttk.LabelFrame(images_frame, text="Imagen Procesada", padding="5")
        processed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        processed_frame.columnconfigure(0, weight=1)
        processed_frame.rowconfigure(0, weight=1)

        self.processed_canvas = tk.Canvas(processed_frame, bg='lightgray', width=400, height=300)
        self.processed_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


        histogram_display_frame = ttk.LabelFrame(right_frame, text="Histogramas", padding="10")
        histogram_display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        histogram_display_frame.columnconfigure(0, weight=1)
        histogram_display_frame.rowconfigure(0, weight=1)


        self.histogram_figure = Figure(figsize=(10, 4), dpi=100)
        self.histogram_ax = self.histogram_figure.add_subplot(111)
        self.histogram_ax.set_title('Selecciona una opción de histograma')
        self.histogram_ax.set_xlabel('Valores de píxel')
        self.histogram_ax.set_ylabel('Frecuencia')
        self.histogram_ax.grid(True, alpha=0.3)


        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_figure, histogram_display_frame)
        self.histogram_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


        self.root.update()

    def delete_image1(self):

        if self.original_image is not None:
            self.original_image = None
            self.processed_image = None
            self.gray_image = None
            self.binary_image = None
            self.display_image(None, self.original_canvas)
            self.display_image(None, self.processed_canvas)
            self.add_info("Imagen 1 eliminada.")
        else:
            messagebox.showinfo("Información", "No hay imagen 1 para eliminar.")

    def delete_image2(self):
        """Eliminar la imagen 2"""
        if self.image2 is not None:
            self.image2 = None
            self.add_info("Imagen 2 eliminada.")
        else:
            messagebox.showinfo("Información", "No hay imagen 2 para eliminar.")

    def delete_all_images(self):
        """Eliminar todas las imágenes"""
        self.original_image = None
        self.processed_image = None
        self.image2 = None
        self.gray_image = None
        self.binary_image = None

        self.display_image(None, self.original_canvas)
        self.display_image(None, self.processed_canvas)

        self.add_info("Todas las imágenes eliminadas.")
        messagebox.showinfo("Éxito", "Todas las imágenes han sido eliminadas.")

    def clear_histogram(self):
        """Limpiar el histograma"""
        self.histogram_ax.clear()
        self.histogram_ax.set_title('Selecciona una opción de histograma')
        self.histogram_ax.set_xlabel('Valores de píxel')
        self.histogram_ax.set_ylabel('Frecuencia')
        self.histogram_ax.grid(True, alpha=0.3)
        self.histogram_canvas.draw()
        self.add_info("Histograma limpiado.")

    def reset_image(self):
        """Restablecer la imagen procesada a la original"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.processed_image, self.processed_canvas)
            self.add_info("Imagen procesada restablecida al estado original.")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen original para restablecer.")

    def get_numeric_input(self, title, prompt, default_value=0):

        while True:
            try:
                value = simpledialog.askfloat(title, prompt, initialvalue=default_value)
                if value is None:  # Usuario canceló
                    return None
                return value
            except ValueError:
                messagebox.showerror("Error", "Por favor ingresa un número válido")

    def load_image1(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen 1",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
                self.display_image(self.original_image, self.original_canvas)
                self.display_image(self.processed_image, self.processed_canvas)
                self.add_info("Imagen 1 cargada correctamente.")
                # Mostrar información de la imagen
                h, w = self.original_image.shape[:2]
                channels = self.original_image.shape[2] if len(self.original_image.shape) == 3 else 1
                self.add_info(f"Dimensiones: {w}x{h}, Canales: {channels}")
            else:
                messagebox.showerror("Error", "No se pudo cargar la imagen.")

    def load_image2(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen 2",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image2 = cv2.imread(file_path)
            if self.image2 is not None:
                self.add_info("Imagen 2 cargada correctamente.")
                # Mostrar información de la imagen 2
                h, w = self.image2.shape[:2]
                channels = self.image2.shape[2] if len(self.image2.shape) == 3 else 1
                self.add_info(f"Imagen 2 - Dimensiones: {w}x{h}, Canales: {channels}")
            else:
                messagebox.showerror("Error", "No se pudo cargar la imagen 2.")

    def display_image(self, image, canvas):

        if image is None:
            canvas.delete("all")
            canvas.create_text(200, 150, text="No hay imagen", fill="black", font=('Arial', 12))
            return

        try:
            # Convertir BGR a RGB para mostrar correctamente
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image

            # Obtener dimensiones del canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Si el canvas no tiene tamaño aún, usar valores por defecto
            if canvas_width <= 1:
                canvas_width = 400
            if canvas_height <= 1:
                canvas_height = 300


            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h) * 0.95  # 95% del espacio disponible
            new_w, new_h = int(w * scale), int(h * scale)

            image_resized = cv2.resize(image_rgb, (new_w, new_h))

            pil_image = Image.fromarray(image_resized)


            if canvas == self.original_canvas:
                self.original_photo = ImageTk.PhotoImage(pil_image)
                photo = self.original_photo
            else:
                self.processed_photo = ImageTk.PhotoImage(pil_image)
                photo = self.processed_photo


            canvas.delete("all")
            x_center = canvas_width // 2
            y_center = canvas_height // 2
            canvas.create_image(x_center, y_center, image=photo, anchor=tk.CENTER)

        except Exception as e:
            canvas.delete("all")
            canvas.create_text(200, 150, text=f"Error: {str(e)}", fill="red", font=('Arial', 10))

    def add_info(self, message):
        self.info_text.insert(tk.END, f"> {message}\n")
        self.info_text.see(tk.END)

    def arithmetic_operation(self, operation):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return


        if operation == 'add':
            value = self.get_numeric_input("Suma", "Ingresa el valor a sumar a cada píxel:", 50)
            if value is None: return
            self.processed_image = cv2.add(self.original_image, value)
            self.add_info(f"Operación: Suma +{value} aplicada.")

        elif operation == 'subtract':
            value = self.get_numeric_input("Resta", "Ingresa el valor a restar a cada píxel:", 50)
            if value is None: return
            self.processed_image = cv2.subtract(self.original_image, value)
            self.add_info(f"Operación: Resta -{value} aplicada.")

        elif operation == 'multiply':
            value = self.get_numeric_input("Multiplicación", "Ingresa el factor de multiplicación:", 1.2)
            if value is None: return
            self.processed_image = cv2.multiply(self.original_image, value)
            self.add_info(f"Operación: Multiplicación x{value} aplicada.")

        elif operation == 'divide':
            value = self.get_numeric_input("División", "Ingresa el divisor:", 2.0)
            if value is None: return
            if value == 0:
                messagebox.showerror("Error", "No se puede dividir por cero.")
                return
            self.processed_image = cv2.divide(self.original_image, value)
            self.add_info(f"Operación: División /{value} aplicada.")

        self.display_image(self.processed_image, self.processed_canvas)

    def logical_operation(self, operation):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga la imagen 1.")
            return

        if operation != 'not' and self.image2 is None:
            messagebox.showwarning("Advertencia", "Para esta operación necesitas cargar la imagen 2.")
            return

        # Asegurar que las imágenes tengan el mismo tamaño
        if operation != 'not':
            # Obtener el tamaño mínimo común
            h1, w1 = self.original_image.shape[:2]
            h2, w2 = self.image2.shape[:2]
            h = min(h1, h2)
            w = min(w1, w2)

            img1_resized = cv2.resize(self.original_image, (w, h))
            img2_resized = cv2.resize(self.image2, (w, h))

            if operation == 'and':
                self.processed_image = cv2.bitwise_and(img1_resized, img2_resized)
                self.add_info("Operación: AND aplicada.")
            elif operation == 'or':
                self.processed_image = cv2.bitwise_or(img1_resized, img2_resized)
                self.add_info("Operación: OR aplicada.")
            elif operation == 'xor':
                self.processed_image = cv2.bitwise_xor(img1_resized, img2_resized)
                self.add_info("Operación: XOR aplicada.")
        else:
            self.processed_image = cv2.bitwise_not(self.original_image)
            self.add_info("Operación: NOT aplicada.")

        self.display_image(self.processed_image, self.processed_canvas)

    def convert_to_grayscale(self):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = self.gray_image
        self.display_image(self.processed_image, self.processed_canvas)
        self.add_info("Imagen convertida a escala de grises.")

    def threshold_image(self):
        """Umbralización con valor por defecto"""
        self.apply_threshold(127)

    def custom_threshold(self):
        """Umbralización con valor personalizado"""
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Si no está en escala de grises, convertir primero
        if len(self.original_image.shape) == 3:
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray = self.gray_image
        else:
            gray = self.original_image

        # Obtener valor de umbral del usuario
        threshold_value = self.get_numeric_input("Umbralización", "Ingresa el valor de umbral (0-255):", 127)
        if threshold_value is None: return

        # Asegurar que esté en el rango correcto
        threshold_value = max(0, min(255, threshold_value))

        self.apply_threshold(threshold_value)

    def apply_threshold(self, threshold_value):
        """Aplicar umbralización con el valor especificado"""
        if self.original_image is None:
            return

        # Si no está en escala de grises, convertir primero
        if len(self.original_image.shape) == 3:
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray = self.gray_image
        else:
            gray = self.original_image

        # Aplicar umbralización
        _, self.binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        self.processed_image = self.binary_image
        self.display_image(self.processed_image, self.processed_canvas)
        self.add_info(f"Umbralización aplicada (umbral={threshold_value}).")

    def connected_components(self, connectivity):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Si no está binarizada, aplicar umbralización primero
        if self.binary_image is None:
            self.custom_threshold()
            if self.binary_image is None:  # Si el usuario canceló
                return

        # Aplicar etiquetado de componentes conexas
        num_labels, labels = cv2.connectedComponents(self.binary_image, connectivity=connectivity)

        # Mostrar resultados
        self.add_info(f"Vecindad {connectivity}: {num_labels - 1} objetos detectados.")

        # Crear imagen de resultados
        labels_normalized = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        self.processed_image = cv2.applyColorMap(labels_normalized, cv2.COLORMAP_JET)
        self.display_image(self.processed_image, self.processed_canvas)

    def contour_objects(self):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Si no está binarizada, aplicar umbralización primero
        if self.binary_image is None:
            self.custom_threshold()
            if self.binary_image is None:  # Si el usuario canceló
                return

        # Encontrar contornos
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crear imagen con contornos
        if len(self.original_image.shape) == 3:
            result_image = self.original_image.copy()
        else:
            result_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)

        # Dibujar contornos y numerar objetos
        for i, contour in enumerate(contours):
            # Dibujar contorno
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)

            # Encontrar centro y colocar número
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(result_image, f'{i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.processed_image = result_image
        self.display_image(self.processed_image, self.processed_canvas)
        self.add_info(f"{len(contours)} objetos contorneados y numerados.")

    def show_histogram(self, mode):
        """Mostrar histograma según el modo especificado"""
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        self.histogram_ax.clear()

        if mode == 'rgb' and len(self.original_image.shape) == 3:
            # Histograma RGB
            colors = ('b', 'g', 'r')
            color_names = ('Azul', 'Verde', 'Rojo')
            for i, (color, name) in enumerate(zip(colors, color_names)):
                histogram = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
                self.histogram_ax.plot(histogram, color=color, label=name, alpha=0.7)
            self.histogram_ax.set_title('Histograma - Imagen Color RGB')
            self.histogram_ax.legend()

        elif mode == 'gray':

            if self.gray_image is None:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.gray_image
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
            self.histogram_ax.plot(histogram, color='black', label='Escala de Grises', linewidth=2)
            self.histogram_ax.set_title('Histograma - Escala de Grises')
            self.histogram_ax.legend()

        elif mode == 'binary':

            if self.binary_image is None:
                messagebox.showwarning("Advertencia", "Primero aplica umbralización.")
                return
            histogram = cv2.calcHist([self.binary_image], [0], None, [256], [0, 256])
            self.histogram_ax.plot(histogram, color='red', label='Imagen Binarizada', linewidth=2)
            self.histogram_ax.set_title('Histograma - Imagen Binarizada')
            self.histogram_ax.legend()

        self.histogram_ax.set_xlabel('Valores de píxel')
        self.histogram_ax.set_ylabel('Frecuencia')
        self.histogram_ax.grid(True, alpha=0.3)
        self.histogram_canvas.draw()

        self.add_info(f"Histograma {mode.upper()} mostrado.")

    def compare_histograms(self):

        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        if self.processed_image is None or np.array_equal(self.original_image, self.processed_image):
            messagebox.showwarning("Advertencia", "No hay imagen procesada para comparar.")
            return

        self.histogram_ax.clear()

        # Histograma original
        if len(self.original_image.shape) == 3:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                histogram = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
                self.histogram_ax.plot(histogram, color=color, linestyle='-', label=f'Original {color.upper()}',
                                       alpha=0.6)
        else:
            histogram = cv2.calcHist([self.original_image], [0], None, [256], [0, 256])
            self.histogram_ax.plot(histogram, color='black', linestyle='-', label='Original', linewidth=2)

        # Histograma procesada
        if len(self.processed_image.shape) == 3:
            colors = ('c', 'm', 'y')
            for i, color in enumerate(colors):
                histogram = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                self.histogram_ax.plot(histogram, color=color, linestyle='--', label=f'Procesada {["B", "G", "R"][i]}',
                                       alpha=0.8)
        else:
            histogram = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
            self.histogram_ax.plot(histogram, color='blue', linestyle='--', label='Procesada', linewidth=2)

        self.histogram_ax.set_title('Comparación de Histogramas - Original vs Procesada')
        self.histogram_ax.set_xlabel('Valores de píxel')
        self.histogram_ax.set_ylabel('Frecuencia')
        self.histogram_ax.legend()
        self.histogram_ax.grid(True, alpha=0.3)
        self.histogram_canvas.draw()

        self.add_info("Comparación de histogramas mostrada.")

    def update_display(self):
        """Actualizar la visualización cuando cambia el tamaño de la ventana"""
        if hasattr(self, 'original_image') and self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas)
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_canvas)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)


    root.after(100, app.update_display)

    root.mainloop()