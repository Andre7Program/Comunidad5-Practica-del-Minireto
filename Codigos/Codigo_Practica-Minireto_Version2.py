import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Minireto. Extracción de Componentes Conexas")
        self.root.geometry("1200x820")
        self.root.configure(bg='#f0f0f0')

        # imágenes y estados
        self.original_image = None
        self.processed_image = None
        self.image2 = None
        self.gray_image = None
        self.binary_image = None

        # referencias PhotoImage (para tkinter)
        self.original_photo = None
        self.processed_photo = None

        # label para mensajes debajo de las imágen
        self.message_label = None

        # objetos matplotlib para histogramas
        self.hist_figure = None
        self.hist_canvas = None
        self.hist_ax_orig = None
        self.hist_ax_proc = None

        self.setup_ui()

    def setup_ui(self):
        # ----------------- título centrado -----------------
        title_frame = ttk.Frame(self.root, padding=(6,6))
        title_frame.pack(fill=tk.X)
        title_label = ttk.Label(title_frame,
                                text="Transformaciones Lógicas y Etiquetado de Componentes Conexas",
                                font=('Arial', 14, 'bold'))
        title_label.pack(pady=(6,4))

        # ----------------- layout principal -----------------
        main_frame = ttk.Frame(self.root, padding=(6,6))
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # LEFT: CONTROLES (restaurado compacto)
        left_frame = ttk.Frame(main_frame, width=320)
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0,6))
        left_frame.grid_propagate(False)
        left_frame.columnconfigure(0, weight=1)

        controls_frame = ttk.LabelFrame(left_frame, text="Controles", padding="8")
        controls_frame.pack(fill=tk.BOTH, expand=True)

        # --- carga / guardado / reset ---
        load_frame = ttk.Frame(controls_frame)
        load_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=4)
        ttk.Button(load_frame, text="Cargar Imagen 1", command=self.load_image1).grid(row=0, column=0, padx=2)
        ttk.Button(load_frame, text="Cargar Imagen 2", command=self.load_image2).grid(row=0, column=1, padx=2)
        ttk.Button(load_frame, text="Guardar Procesada", command=self.save_processed_image).grid(row=0, column=2, padx=2)
        ttk.Button(load_frame, text="Reset Procesada", command=self.reset_image).grid(row=0, column=3, padx=2)
        ttk.Button(load_frame, text="Eliminar Imagen 1", command=self.delete_image1).grid(row=1, column=0, padx=2, pady=4)
        ttk.Button(load_frame, text="Eliminar Imagen 2", command=self.delete_image2).grid(row=1, column=1, padx=2, pady=4)
        ttk.Button(load_frame, text="Eliminar Todas", command=self.delete_all_images).grid(row=1, column=2, padx=2, pady=4)

        # --- Operaciones aritméticas con escalar
        arith_scalar_frame = ttk.LabelFrame(controls_frame, text="Operaciones Aritméticas con Escalar", padding=6)
        arith_scalar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(arith_scalar_frame, text="Suma", command=lambda: self.arithmetic_operation_scalar('add')).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(arith_scalar_frame, text="Resta", command=lambda: self.arithmetic_operation_scalar('subtract')).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(arith_scalar_frame, text="Multiplicación", command=lambda: self.arithmetic_operation_scalar('multiply')).grid(row=0, column=2, padx=2, pady=3)

        # --- Operaciones aritméticas entre imágenes
        arith_images_frame = ttk.LabelFrame(controls_frame, text="Operaciones Aritméticas entre Imágenes", padding=6)
        arith_images_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(arith_images_frame, text="Suma", command=lambda: self.arithmetic_operation_images('add')).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(arith_images_frame, text="Resta", command=lambda: self.arithmetic_operation_images('subtract')).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(arith_images_frame, text="Multiplicación", command=lambda: self.arithmetic_operation_images('multiply')).grid(row=0, column=2, padx=2, pady=3)

        # --- Operaciones lógicas ---
        logic_frame = ttk.LabelFrame(controls_frame, text="Operaciones Lógicas", padding=6)
        logic_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(logic_frame, text="AND", command=lambda: self.logical_operation('and')).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(logic_frame, text="OR", command=lambda: self.logical_operation('or')).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(logic_frame, text="XOR", command=lambda: self.logical_operation('xor')).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(logic_frame, text="NOT", command=lambda: self.logical_operation('not')).grid(row=0, column=3, padx=2, pady=3)

        # --- Preprocesamiento ---
        preprocess_frame = ttk.LabelFrame(controls_frame, text="Preprocesamiento", padding=6)
        preprocess_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(preprocess_frame, text="Escala de Grises", command=self.convert_to_grayscale).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(preprocess_frame, text="Umbralizar", command=self.threshold_image).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(preprocess_frame, text="Umbral Personalizado", command=self.custom_threshold).grid(row=0, column=2, padx=2, pady=3)

        # --- Etiquetado de componentes ---
        labeling_frame = ttk.LabelFrame(controls_frame, text="Etiquetado de Componentes Conexas", padding=6)
        labeling_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(labeling_frame, text="Vecindad 4", command=lambda: self.connected_components(4)).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(labeling_frame, text="Vecindad 8", command=lambda: self.connected_components(8)).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(labeling_frame, text="Contornear Objetos", command=self.contour_objects).grid(row=0, column=2, padx=2, pady=3)


        # ---- visualización controles -----------------
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        right_frame.columnconfigure(0, weight=1)
        # filas: 0 imágenes, 1 mensaje, 2 histogramas
        right_frame.rowconfigure(0, weight=3)
        right_frame.rowconfigure(1, weight=0)
        right_frame.rowconfigure(2, weight=1)

        # IMÁGENES: originales y procesadas lado a lado
        images_frame = ttk.Frame(right_frame, padding=(0,0,0,2))
        images_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)

        original_frame = ttk.LabelFrame(images_frame, text="Imagen Original", padding=6)
        original_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0,6))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        self.original_canvas = tk.Canvas(original_frame, bg='lightgray', width=540, height=300)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        processed_frame = ttk.LabelFrame(images_frame, text="Imagen Procesada", padding=6)
        processed_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        processed_frame.columnconfigure(0, weight=1)
        processed_frame.rowconfigure(0, weight=1)
        self.processed_canvas = tk.Canvas(processed_frame, bg='lightgray', width=540, height=300)
        self.processed_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.message_label = ttk.Label(right_frame, text="Mensajes aparecerán aquí.", anchor=tk.CENTER, font=('Arial', 10), foreground='blue')
        self.message_label.grid(row=1, column=0, pady=(6,3), sticky=(tk.W, tk.E))

        hist_frame = ttk.LabelFrame(right_frame, text="Histogramas (Original | Procesada)", padding=6)
        hist_frame.grid(row=2, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(2,4))
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.rowconfigure(0, weight=1)

        self.hist_figure = Figure(figsize=(10, 2.8), dpi=100)
        self.hist_ax_orig = self.hist_figure.add_subplot(1,2,1)
        self.hist_ax_proc = self.hist_figure.add_subplot(1,2,2)
        self.hist_ax_orig.set_title('Original')
        self.hist_ax_proc.set_title('Procesada')
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.update()

    def set_message(self, text):
        if self.message_label is not None:
            self.message_label.config(text=text)
        print(text)

    def save_processed_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar.")
            return
        file_path = filedialog.asksaveasfilename(title="Guardar imagen procesada", defaultextension=".png",
                                                 filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("BMP","*.bmp"),("TIFF","*.tiff")])
        if not file_path:
            return
        success = cv2.imwrite(file_path, self.processed_image)
        if success:
            self.set_message(f"Imagen procesada guardada en: {file_path}")
            messagebox.showinfo("Guardado", "Imagen procesada guardada correctamente.")
        else:
            messagebox.showerror("Error", "No se pudo guardar la imagen.")

    def get_numeric_input(self, title, prompt, default_value=0):
        try:
            value = simpledialog.askfloat(title, prompt, initialvalue=default_value)
            return value
        except:
            return None

    # ----------------- carga / eliminación / display -----------------
    def load_image1(self):
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen 1",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen.")
                return
            self.original_image = img
            self.processed_image = img.copy()
            self.gray_image = None
            self.binary_image = None
            self.display_image(self.original_image, self.original_canvas)
            self.display_image(self.processed_image, self.processed_canvas)
            self.set_message("Imagen 1 cargada correctamente.")
            self.show_histogram_auto()

    def load_image2(self):
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen 2",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen 2.")
                return
            self.image2 = img
            self.set_message("Imagen 2 cargada correctamente.")

    def delete_image1(self):
        if self.original_image is not None:
            self.original_image = None
            self.processed_image = None
            self.gray_image = None
            self.binary_image = None
            self.display_image(None, self.original_canvas)
            self.display_image(None, self.processed_canvas)
            self.clear_histogram()
            self.set_message("Imagen 1 eliminada.")
        else:
            messagebox.showinfo("Información", "No hay imagen 1 para eliminar.")

    def delete_image2(self):
        if self.image2 is not None:
            self.image2 = None
            self.set_message("Imagen 2 eliminada.")
        else:
            messagebox.showinfo("Información", "No hay imagen 2 para eliminar.")

    def delete_all_images(self):
        self.original_image = None
        self.processed_image = None
        self.image2 = None
        self.gray_image = None
        self.binary_image = None
        self.display_image(None, self.original_canvas)
        self.display_image(None, self.processed_canvas)
        self.clear_histogram()
        self.set_message("Todas las imágenes eliminadas.")

    def reset_image(self):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "No hay imagen original para restablecer.")
            return
        self.processed_image = self.original_image.copy()
        self.gray_image = None
        self.binary_image = None
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()
        self.set_message("Imagen procesada restablecida al estado original.")

    def display_image(self, image, canvas):
        canvas.delete("all")
        if image is None:
            canvas.create_text(canvas.winfo_reqwidth()//2, canvas.winfo_reqheight()//2,
                               text="No hay imagen", fill="black", font=('Arial', 12))
            return
        try:
            if image.ndim == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else canvas.winfo_reqwidth()
            canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else canvas.winfo_reqheight()
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h) * 0.95
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(image_rgb, (new_w, new_h))
            pil = ImageTk.PhotoImage(Image.fromarray(resized))
            if canvas == self.original_canvas:
                self.original_photo = pil
            else:
                self.processed_photo = pil
            x_center = canvas_width // 2
            y_center = canvas_height // 2
            canvas.create_image(x_center, y_center, image=pil, anchor=tk.CENTER)
        except Exception as e:
            canvas.create_text(200,120, text=f"Error: {e}", fill='red')

    # ----------------- operaciones aritméticas -----------------
    def arithmetic_operation_scalar(self, operation):
        src = self.processed_image if self.processed_image is not None else self.original_image
        if src is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # si la imagen es binaria: permitir entrada pero no cambiar visualmente
        if self.binary_image is not None and self.processed_image is not None and np.array_equal(self.processed_image, self.binary_image):
            if operation == 'add':
                value = self.get_numeric_input("Suma con Escalar", "Ingresa el valor a sumar a cada píxel:", 50)
                if value is None: return
                self.set_message(f"Suma con escalar +{value} solicitada sobre imagen binarizada (sin cambios visibles).")
            elif operation == 'subtract':
                value = self.get_numeric_input("Resta con Escalar", "Ingresa el valor a restar a cada píxel:", 50)
                if value is None: return
                self.set_message(f"Resta con escalar -{value} solicitada sobre imagen binarizada (sin cambios visibles).")
            elif operation == 'multiply':
                value = self.get_numeric_input("Multiplicación con Escalar", "Ingresa el factor de multiplicación:", 1.2)
                if value is None: return
                self.set_message(f"Multiplicación con escalar x{value} solicitada sobre imagen binarizada (sin cambios visibles).")
            self.display_image(self.processed_image, self.processed_canvas)
            self.show_histogram_auto()
            return

        # pedir valor y aplicar sobre imagen original
        if operation == 'add':
            value = self.get_numeric_input("Suma con Escalar", "Ingresa el valor a sumar a cada píxel:", 50)
            if value is None: return
            self.processed_image = cv2.add(src, value)
            self.set_message(f"Suma con escalar +{value} aplicada.")
        elif operation == 'subtract':
            value = self.get_numeric_input("Resta con Escalar", "Ingresa el valor a restar a cada píxel:", 50)
            if value is None: return
            self.processed_image = cv2.subtract(src, value)
            self.set_message(f"Resta con escalar -{value} aplicada.")
        elif operation == 'multiply':
            value = self.get_numeric_input("Multiplicación con Escalar", "Ingresa el factor de multiplicación:", 1.2)
            if value is None: return
            res = cv2.multiply(src.astype(np.float32), float(value))
            self.processed_image = np.clip(res,0,255).astype(np.uint8)
            self.set_message(f"Multiplicación con escalar x{value} aplicada.")
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()

    def arithmetic_operation_images(self, operation):
        imgA = self.processed_image if self.processed_image is not None else self.original_image
        if imgA is None:
            messagebox.showwarning("Advertencia", "Primero carga la imagen 1.")
            return
        if self.image2 is None:
            messagebox.showwarning("Advertencia", "Para esta operación necesitas cargar la imagen 2.")
            return

        # si binaria
        if self.binary_image is not None and self.processed_image is not None and np.array_equal(self.processed_image, self.binary_image):
            self.set_message("Operación aritmética entre imágenes solicitada sobre imagen binarizada (sin cambios visibles).")
            self.display_image(self.processed_image, self.processed_canvas)
            self.show_histogram_auto()
            return

        h1,w1 = imgA.shape[:2]
        h2,w2 = self.image2.shape[:2]
        h,w = min(h1,h2), min(w1,w2)
        A = cv2.resize(imgA, (w,h))
        B = cv2.resize(self.image2, (w,h))

        A_gray = (A.ndim == 2)
        B_gray = (B.ndim == 2)
        if A_gray and not B_gray:
            B_proc = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
            A_proc = A
        elif not A_gray and B_gray:
            B_proc = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
            A_proc = A
        else:
            A_proc = A
            B_proc = B

        try:
            if operation == 'add':
                result = cv2.add(A_proc, B_proc); self.set_message("Suma de imágenes aplicada.")
            elif operation == 'subtract':
                result = cv2.subtract(A_proc, B_proc); self.set_message("Resta de imágenes aplicada.")
            elif operation == 'multiply':
                result = cv2.multiply(A_proc, B_proc); self.set_message("Multiplicación de imágenes aplicada.")
            else:
                messagebox.showerror("Error", "Operación desconocida.")
                return
            if result.dtype != np.uint8:
                result = np.clip(result,0,255).astype(np.uint8)
            self.processed_image = result
            self.display_image(self.processed_image, self.processed_canvas)
            self.show_histogram_auto()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo realizar la operación: {e}")

    # ----------------- operaciones lógicas -----------------
    def logical_operation(self, operation):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga la imagen 1.")
            return
        if operation != 'not' and self.image2 is None:
            messagebox.showwarning("Advertencia", "Para esta operación necesitas cargar la imagen 2.")
            return

        if operation != 'not':
            h1,w1 = self.original_image.shape[:2]
            h2,w2 = self.image2.shape[:2]
            h,w = min(h1,h2), min(w1,w2)
            A = cv2.resize(self.original_image, (w,h))
            B = cv2.resize(self.image2, (w,h))
            if operation == 'and':
                res = cv2.bitwise_and(A,B); self.set_message("Operación: AND aplicada.")
            elif operation == 'or':
                res = cv2.bitwise_or(A,B); self.set_message("Operación: OR aplicada.")
            elif operation == 'xor':
                res = cv2.bitwise_xor(A,B); self.set_message("Operación: XOR aplicada.")
            else:
                messagebox.showerror("Error", "Operación desconocida.")
                return
            self.processed_image = res
        else:
            self.processed_image = cv2.bitwise_not(self.original_image)
            self.set_message("Operación: NOT aplicada.")
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()

    # ----------------- preprocesamiento -----------------
    def convert_to_grayscale(self):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = self.gray_image
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()
        self.set_message("Imagen convertida a escala de grises.")

    def threshold_image(self):
        self.apply_threshold(127)

    def custom_threshold(self):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        if len(self.original_image.shape) == 3:
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray = self.gray_image
        else:
            gray = self.original_image
        threshold_value = self.get_numeric_input("Umbralización", "Ingresa el valor de umbral (0-255):", 127)
        if threshold_value is None: return
        threshold_value = max(0, min(255, threshold_value))
        self.apply_threshold(threshold_value)

    def apply_threshold(self, threshold_value):
        if self.original_image is None:
            return
        if len(self.original_image.shape) == 3:
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray = self.gray_image
        else:
            gray = self.original_image
        _, self.binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        self.processed_image = self.binary_image
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()
        self.set_message(f"Umbralización aplicada (umbral={threshold_value}).")

    def connected_components(self, connectivity):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        if self.binary_image is None:
            self.custom_threshold()
            if self.binary_image is None:
                return
        num_labels, labels = cv2.connectedComponents(self.binary_image, connectivity=connectivity)
        self.set_message(f"Vecindad {connectivity}: {num_labels-1} objetos detectados.")
        labels_normalized = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        self.processed_image = cv2.applyColorMap(labels_normalized, cv2.COLORMAP_JET)
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()

    def contour_objects(self):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        if self.binary_image is None:
            self.custom_threshold()
            if self.binary_image is None:
                return
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(self.original_image.shape) == 3:
            result_image = self.original_image.copy()
        else:
            result_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours):
            cv2.drawContours(result_image, [contour], -1, (0,255,0), 2)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.putText(result_image, f'{i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        self.processed_image = result_image
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()
        self.set_message(f"{len(contours)} objetos contorneados y numerados.")

    # ----------------- histogramas -----------------
    def clear_histogram(self):
        self.hist_ax_orig.clear()
        self.hist_ax_proc.clear()
        self.hist_ax_orig.set_title('Original')
        self.hist_ax_proc.set_title('Procesada')
        self.hist_canvas.draw()

    def show_histogram(self, mode):
        """Dibuja histogramas según modo ('rgb','gray','binary')"""
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        self.hist_ax_orig.clear()
        self.hist_ax_proc.clear()

        # original
        orig = self.original_image
        if mode == 'rgb' and orig.ndim == 3:
            colors = ('b','g','r'); names = ('Azul','Verde','Rojo')
            for i,col in enumerate(colors):
                hist = cv2.calcHist([orig], [i], None, [256], [0,256])
                self.hist_ax_orig.plot(hist, label=names[i])
            self.hist_ax_orig.set_title('Original - RGB'); self.hist_ax_orig.legend(fontsize='small')
        else:
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if orig.ndim==3 else orig
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            self.hist_ax_orig.plot(hist, label='Grises')
            self.hist_ax_orig.set_title('Original - Grises'); self.hist_ax_orig.legend(fontsize='small')

        # procesada
        proc = self.processed_image
        if proc is None:
            self.hist_ax_proc.set_title('Procesada - (vacía)')
        else:
            if mode == 'binary':
                proc_gray = proc if proc.ndim==2 else cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                hist_p = cv2.calcHist([proc_gray], [0], None, [256], [0,256])
                self.hist_ax_proc.plot(hist_p, label='Binarizada'); self.hist_ax_proc.set_title('Procesada - Binarizada')
                self.hist_ax_proc.legend(fontsize='small')
            elif mode == 'rgb' and proc.ndim == 3:
                colors = ('b','g','r'); names = ('Azul','Verde','Rojo')
                for i,col in enumerate(colors):
                    hist = cv2.calcHist([proc], [i], None, [256], [0,256])
                    self.hist_ax_proc.plot(hist, label=names[i])
                self.hist_ax_proc.set_title('Procesada - RGB'); self.hist_ax_proc.legend(fontsize='small')
            else:
                pgray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY) if proc.ndim==3 else proc
                hist_p = cv2.calcHist([pgray], [0], None, [256], [0,256])
                self.hist_ax_proc.plot(hist_p, label='Grises'); self.hist_ax_proc.set_title('Procesada - Grises')
                self.hist_ax_proc.legend(fontsize='small')

        self.hist_ax_orig.set_xlabel('Valor de píxel'); self.hist_ax_orig.set_ylabel('Frecuencia')
        self.hist_ax_proc.set_xlabel('Valor de píxel'); self.hist_ax_proc.set_ylabel('Frecuencia')
        self.hist_ax_orig.grid(True, alpha=0.3); self.hist_ax_proc.grid(True, alpha=0.3)
        self.hist_canvas.draw()
        self.set_message(f"Histograma mostrado (modo={mode}).")

    def show_histogram_auto(self):
        if self.original_image is None:
            self.clear_histogram()
            return
        # modo para original
        mode_orig = 'rgb' if self.original_image.ndim == 3 else 'gray'
        # modo para procesada
        if self.processed_image is None:
            mode_proc = None
        else:
            if self.binary_image is not None and np.array_equal(self.processed_image, self.binary_image):
                mode_proc = 'binary'
            else:
                mode_proc = 'rgb' if self.processed_image.ndim == 3 else 'gray'

        self.hist_ax_orig.clear(); self.hist_ax_proc.clear()

        # original
        orig = self.original_image
        if mode_orig == 'rgb':
            colors = ('b','g','r'); names = ('Azul','Verde','Rojo')
            for i,col in enumerate(colors):
                hist = cv2.calcHist([orig], [i], None, [256], [0,256])
                self.hist_ax_orig.plot(hist, label=names[i])
            self.hist_ax_orig.set_title('Original - RGB'); self.hist_ax_orig.legend(fontsize='small')
        else:
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if orig.ndim==3 else orig
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            self.hist_ax_orig.plot(hist, label='Grises')
            self.hist_ax_orig.set_title('Original - Grises'); self.hist_ax_orig.legend(fontsize='small')

        # procesada
        proc = self.processed_image
        if proc is None:
            self.hist_ax_proc.set_title('Procesada - (vacía)')
        else:
            if mode_proc == 'binary':
                proc_gray = proc if proc.ndim==2 else cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                hist_p = cv2.calcHist([proc_gray], [0], None, [256], [0,256])
                self.hist_ax_proc.plot(hist_p, label='Binarizada'); self.hist_ax_proc.set_title('Procesada - Binarizada')
                self.hist_ax_proc.legend(fontsize='small')
            elif mode_proc == 'rgb':
                colors = ('b','g','r'); names = ('Azul','Verde','Rojo')
                for i,col in enumerate(colors):
                    hist = cv2.calcHist([proc], [i], None, [256], [0,256])
                    self.hist_ax_proc.plot(hist, label=names[i])
                self.hist_ax_proc.set_title('Procesada - RGB'); self.hist_ax_proc.legend(fontsize='small')
            else:
                pgray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY) if proc.ndim==3 else proc
                hist_p = cv2.calcHist([pgray], [0], None, [256], [0,256])
                self.hist_ax_proc.plot(hist_p, label='Grises'); self.hist_ax_proc.set_title('Procesada - Grises')
                self.hist_ax_proc.legend(fontsize='small')

        self.hist_ax_orig.set_xlabel('Valor de píxel'); self.hist_ax_orig.set_ylabel('Frecuencia')
        self.hist_ax_proc.set_xlabel('Valor de píxel'); self.hist_ax_proc.set_ylabel('Frecuencia')
        self.hist_ax_orig.grid(True, alpha=0.3); self.hist_ax_proc.grid(True, alpha=0.3)
        self.hist_canvas.draw()
        self.set_message("Histogramas actualizados.")

    def update_display(self):
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas)
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_canvas)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.after(120, app.update_display)
    root.mainloop()
