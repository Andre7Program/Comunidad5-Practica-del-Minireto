import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats
from collections import Counter
from scipy.signal import find_peaks
import math


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Minireto - Segmentación y Ajuste de Brillo")
        self.root.geometry("1600x900")  # Ventana más ancha
        self.root.configure(bg='#f0f0f0')

        # imágenes y estados
        self.original_image = None
        self.processed_image = None
        self.image2 = None
        self.gray_image = None
        self.binary_image = None

        # Historial de procesamiento
        self.processing_history = []

        # referencias PhotoImage (para tkinter)
        self.original_photo = None
        self.processed_photo = None

        # label para mensajes debajo de las imágenes
        self.message_label = None

        # objetos matplotlib para histogramas
        self.hist_figure = None
        self.hist_canvas = None
        self.hist_ax_orig = None
        self.hist_ax_proc = None

        self.setup_ui()

    def setup_ui(self):
        # ----------------- título centrado -----------------
        title_frame = ttk.Frame(self.root, padding=(6, 6))
        title_frame.pack(fill=tk.X)
        title_label = ttk.Label(title_frame,
                                text="Minireto - Segmentación y Ajuste de Brillo",
                                font=('Arial', 14, 'bold'))
        title_label.pack(pady=(6, 4))

        # ----------------- layout principal -----------------
        main_frame = ttk.Frame(self.root, padding=(6, 6))
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)


        left_frame = ttk.Frame(main_frame, width=500)
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0, 6))
        left_frame.grid_propagate(False)
        left_frame.columnconfigure(0, weight=1)

        # Crear scrollbar para controles
        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        controls_frame = ttk.LabelFrame(scrollable_frame, text="Controles", padding="8")
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

        # --- Control de Historial ---
        history_frame = ttk.LabelFrame(controls_frame, text="Control de Procesamiento", padding=6)
        history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(history_frame, text="Deshacer Último", command=self.undo_last_processing).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(history_frame, text="Ver Historial", command=self.show_processing_history).grid(row=0, column=1, padx=2, pady=3)

        # --- Generación de Ruido ---
        noise_frame = ttk.LabelFrame(controls_frame, text="Generación de Ruido", padding=6)
        noise_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(noise_frame, text="Ruido Sal y Pimienta", command=self.add_salt_pepper_noise).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(noise_frame, text="Ruido Gaussiano", command=self.add_gaussian_noise).grid(row=0, column=1, padx=2, pady=3)

        # --- Filtros Paso Bajas (Lineales) ---
        lowpass_frame = ttk.LabelFrame(controls_frame, text="Filtros Paso Bajas (Lineales)", padding=6)
        lowpass_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(lowpass_frame, text="Filtro Promediador", command=self.apply_average_filter).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(lowpass_frame, text="Promediador Pesado", command=self.apply_weighted_average_filter).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(lowpass_frame, text="Filtro Gaussiano", command=self.apply_gaussian_filter).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(lowpass_frame, text="Filtro Bilateral", command=self.apply_bilateral_filter).grid(row=1, column=0, padx=2, pady=3)

        # --- Filtros No Lineales ---
        nonlinear_frame = ttk.LabelFrame(controls_frame, text="Filtros No Lineales", padding=6)
        nonlinear_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(nonlinear_frame, text="Filtro Mediana", command=self.apply_median_filter).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(nonlinear_frame, text="Filtro Moda", command=self.apply_mode_filter).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(nonlinear_frame, text="Filtro Máximo", command=self.apply_max_filter).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(nonlinear_frame, text="Filtro Mínimo", command=self.apply_min_filter).grid(row=1, column=0, padx=2, pady=3)

        # --- Filtros Paso Altas (Detección de Bordes) ---
        highpass_frame = ttk.LabelFrame(controls_frame, text="Filtros Paso Altas (Bordes)", padding=6)
        highpass_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(highpass_frame, text="Sobel", command=self.apply_sobel).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(highpass_frame, text="Prewitt", command=self.apply_prewitt).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(highpass_frame, text="Roberts", command=self.apply_roberts).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(highpass_frame, text="Canny", command=self.apply_canny).grid(row=1, column=0, padx=2, pady=3)
        ttk.Button(highpass_frame, text="Kirsch", command=self.apply_kirsch).grid(row=1, column=1, padx=2, pady=3)
        ttk.Button(highpass_frame, text="Laplaciano", command=self.apply_laplacian).grid(row=1, column=2, padx=2, pady=3)

        # --- Máscaras Direccionales ---
        directional_frame = ttk.LabelFrame(controls_frame, text="Máscaras Direccionales", padding=6)
        directional_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(directional_frame, text="Líneas Horizontales", command=lambda: self.apply_directional_filter('horizontal')).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(directional_frame, text="Líneas Verticales", command=lambda: self.apply_directional_filter('vertical')).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(directional_frame, text="Diagonal Principal", command=lambda: self.apply_directional_filter('diagonal_main')).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(directional_frame, text="Diagonal Secundaria", command=lambda: self.apply_directional_filter('diagonal_secondary')).grid(row=1, column=0, padx=2, pady=3)

        # --- Operaciones aritméticas con escalar
        arith_scalar_frame = ttk.LabelFrame(controls_frame, text="Operaciones Aritméticas con Escalar", padding=6)
        arith_scalar_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(arith_scalar_frame, text="Suma", command=lambda: self.arithmetic_operation_scalar('add')).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(arith_scalar_frame, text="Resta", command=lambda: self.arithmetic_operation_scalar('subtract')).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(arith_scalar_frame, text="Multiplicación", command=lambda: self.arithmetic_operation_scalar('multiply')).grid(row=0, column=2, padx=2, pady=3)

        # --- Operaciones aritméticas entre imágenes
        arith_images_frame = ttk.LabelFrame(controls_frame, text="Operaciones Aritméticas entre Imágenes", padding=6)
        arith_images_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(arith_images_frame, text="Suma", command=lambda: self.arithmetic_operation_images('add')).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(arith_images_frame, text="Resta", command=lambda: self.arithmetic_operation_images('subtract')).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(arith_images_frame, text="Multiplicación", command=lambda: self.arithmetic_operation_images('multiply')).grid(row=0, column=2, padx=2, pady=3)

        # --- Operaciones lógicas ---
        logic_frame = ttk.LabelFrame(controls_frame, text="Operaciones Lógicas", padding=6)
        logic_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(logic_frame, text="AND", command=lambda: self.logical_operation('and')).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(logic_frame, text="OR", command=lambda: self.logical_operation('or')).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(logic_frame, text="XOR", command=lambda: self.logical_operation('xor')).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(logic_frame, text="NOT", command=lambda: self.logical_operation('not')).grid(row=0, column=3, padx=2, pady=3)

        # --- Preprocesamiento ---
        preprocess_frame = ttk.LabelFrame(controls_frame, text="Preprocesamiento", padding=6)
        preprocess_frame.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(preprocess_frame, text="Escala de Grises", command=self.convert_to_grayscale).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(preprocess_frame, text="Umbralizar", command=self.threshold_image).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(preprocess_frame, text="Umbral Personalizado", command=self.custom_threshold).grid(row=0, column=2, padx=2, pady=3)

        # --- Etiquetado de componentes ---
        labeling_frame = ttk.LabelFrame(controls_frame, text="Etiquetado de Componentes Conexas", padding=6)
        labeling_frame.grid(row=11, column=0, sticky=(tk.W, tk.E), pady=6)
        ttk.Button(labeling_frame, text="Vecindad 4", command=lambda: self.connected_components(4)).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(labeling_frame, text="Vecindad 8", command=lambda: self.connected_components(8)).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(labeling_frame, text="Contornear Objetos", command=self.contour_objects).grid(row=0, column=2, padx=2, pady=3)

        # --- TÉCNICAS DE SEGMENTACIÓN POR UMBRALADO ---
        segmentation_frame = ttk.LabelFrame(controls_frame, text="Técnicas de Segmentación", padding=6)
        segmentation_frame.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=6)

        ttk.Button(segmentation_frame, text="Otsu", command=self.apply_otsu).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(segmentation_frame, text="Kapur", command=self.apply_kapur).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(segmentation_frame, text="Mínimo Histograma", command=self.apply_minimo_histograma).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(segmentation_frame, text="Media", command=self.apply_media).grid(row=0, column=3, padx=2, pady=3)
        ttk.Button(segmentation_frame, text="Multiumbral", command=self.apply_multiumbral).grid(row=1, column=0, padx=2, pady=3)
        ttk.Button(segmentation_frame, text="Umbral Banda", command=self.apply_umbral_banda).grid(row=1, column=1, padx=2, pady=3)

        # --- TÉCNICAS DE AJUSTE DE BRILLO ---
        brightness_frame = ttk.LabelFrame(controls_frame, text="Ajuste de Brillo y Contraste", padding=6)
        brightness_frame.grid(row=13, column=0, sticky=(tk.W, tk.E), pady=6)

        ttk.Button(brightness_frame, text="Equalización Uniforme", command=self.equalizacion_uniforme).grid(row=0, column=0, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Equalización Exponencial", command=self.equalizacion_exponencial).grid(row=0, column=1, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Equalización Rayleigh", command=self.equalizacion_rayleigh).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Equalización Hipercúbica", command=self.equalizacion_hipercubica).grid(row=1, column=0, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Equalización Logarítmica", command=self.equalizacion_logaritmica).grid(row=1, column=1, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Función Potencia", command=self.funcion_potencia).grid(row=1, column=2, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Corrección Gamma", command=self.correccion_gamma).grid(row=2, column=0, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Desplazamiento Histograma", command=self.desplazamiento_histograma).grid(row=2, column=1, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Expansión Histograma", command=self.expansion_histograma).grid(row=2, column=2, padx=2, pady=3)
        ttk.Button(brightness_frame, text="Contracción Histograma", command=self.contraccion_histograma).grid(row=2, column=3, padx=2, pady=3)

        # ---- visualización controles -----------------
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        right_frame.columnconfigure(0, weight=1)
        # filas: 0 imágenes, 1 mensaje, 2 histogramas
        right_frame.rowconfigure(0, weight=3)
        right_frame.rowconfigure(1, weight=0)
        right_frame.rowconfigure(2, weight=1)

        # IMÁGENES: originales y procesadas lado a lado
        images_frame = ttk.Frame(right_frame, padding=(0, 0, 0, 2))
        images_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)

        original_frame = ttk.LabelFrame(images_frame, text="Imagen Original", padding=6)
        original_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0, 6))
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

        self.message_label = ttk.Label(right_frame, text="Mensajes aparecerán aquí.", anchor=tk.CENTER,
                                       font=('Arial', 10), foreground='blue')
        self.message_label.grid(row=1, column=0, pady=(6, 3), sticky=(tk.W, tk.E))

        hist_frame = ttk.LabelFrame(right_frame, text="Histogramas (Original | Procesada)", padding=6)
        hist_frame.grid(row=2, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(2, 4))
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.rowconfigure(0, weight=1)

        self.hist_figure = Figure(figsize=(10, 2.8), dpi=100)
        self.hist_ax_orig = self.hist_figure.add_subplot(1, 2, 1)
        self.hist_ax_proc = self.hist_figure.add_subplot(1, 2, 2)
        self.hist_ax_orig.set_title('Original')
        self.hist_ax_proc.set_title('Procesada')
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.update()

    def get_working_image(self):
        """Obtiene la imagen actual para procesar (siempre la procesada si existe)"""
        if self.processed_image is not None:
            return self.processed_image.copy()
        elif self.original_image is not None:
            return self.original_image.copy()
        else:
            return None

    def update_processed_image(self, new_image, operation_name):
        """Actualiza la imagen procesada y guarda en el historial"""
        if new_image is not None:
            self.processed_image = new_image
            self.save_processing_state(operation_name)
            self.display_image(self.processed_image, self.processed_canvas)
            self.show_histogram_auto()
            self.set_message(f"{operation_name} aplicado a imagen procesada")

    # ==================== NUEVAS FUNCIONES DE SEGMENTACIÓN ====================

    def apply_otsu(self):
        """Aplica el método de Otsu para segmentación"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Aplicar Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.binary_image = thresh
        self.update_processed_image(thresh, "Segmentación - Otsu")

    def apply_kapur(self):
        """Aplica el método de entropía de Kapur para segmentación"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        def entropia_kapur(histograma, total_pixeles):
            max_entropia = -1
            umbral_optimo = 0

            for t in range(1, 255):
                # Dividir el histograma en dos clases
                clase1 = histograma[:t]
                clase2 = histograma[t:]

                # Probabilidades
                p1 = np.sum(clase1) / total_pixeles
                p2 = np.sum(clase2) / total_pixeles

                # Evitar divisiones por cero
                if p1 == 0 or p2 == 0:
                    continue

                # Entropias
                entropia1 = -np.sum((clase1 / np.sum(clase1)) * np.log(clase1 / np.sum(clase1) + 1e-10))
                entropia2 = -np.sum((clase2 / np.sum(clase2)) * np.log(clase2 / np.sum(clase2) + 1e-10))

                # Entropia total
                entropia_total = p1 * entropia1 + p2 * entropia2

                if entropia_total > max_entropia:
                    max_entropia = entropia_total
                    umbral_optimo = t

            return umbral_optimo

        # Calcular histograma
        histograma, _ = np.histogram(gray, bins=256, range=(0, 256))
        total_pixeles = gray.size

        # Calcular umbral usando Kapur
        umbral_kapur = entropia_kapur(histograma, total_pixeles)

        # Aplicar umbral
        thresh = (gray > umbral_kapur).astype(np.uint8) * 255

        self.binary_image = thresh
        self.update_processed_image(thresh, f"Segmentación - Kapur (Umbral: {umbral_kapur})")

    def apply_minimo_histograma(self):
        """Aplica segmentación usando el mínimo entre picos del histograma"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Calcular histograma
        histograma, _ = np.histogram(gray, bins=256, range=(0, 256))

        # Encontrar picos en el histograma
        picos, _ = find_peaks(histograma, distance=20, height=100)

        if len(picos) >= 2:
            # Encontrar mínimo entre los dos primeros picos
            minimo = np.argmin(histograma[picos[0]:picos[1]]) + picos[0]
        else:
            # Si no hay suficientes picos, usar Otsu como fallback
            _, minimo = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Aplicar umbral
        thresh = (gray > minimo).astype(np.uint8) * 255

        self.binary_image = thresh
        self.update_processed_image(thresh, f"Segmentación - Mínimo Histograma (Umbral: {minimo})")

    def apply_media(self):
        """Aplica segmentación usando la media como umbral"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Calcular media
        umbral_media = np.mean(gray)

        # Aplicar umbral
        thresh = (gray >= umbral_media).astype(np.uint8) * 255

        self.binary_image = thresh
        self.update_processed_image(thresh, f"Segmentación - Media (Umbral: {umbral_media:.2f})")

    def apply_multiumbral(self):
        """Aplica segmentación con múltiples umbrales"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Pedir umbrales al usuario
        T1 = simpledialog.askinteger("Multiumbral", "Primer umbral T1:", initialvalue=80)
        T2 = simpledialog.askinteger("Multiumbral", "Segundo umbral T2:", initialvalue=150)

        if T1 is None or T2 is None:
            return

        # Segmentación con dos umbrales (tres categorías)
        resultado = np.zeros_like(gray)
        resultado[gray < T1] = 0
        resultado[(gray >= T1) & (gray < T2)] = 127
        resultado[gray >= T2] = 255

        self.update_processed_image(resultado, f"Segmentación - Multiumbral (T1={T1}, T2={T2})")

    def apply_umbral_banda(self):
        """Aplica segmentación por umbral banda"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Pedir umbrales al usuario
        T1 = simpledialog.askinteger("Umbral Banda", "Umbral inferior T1:", initialvalue=80)
        T2 = simpledialog.askinteger("Umbral Banda", "Umbral superior T2:", initialvalue=150)

        if T1 is None or T2 is None:
            return

        # Segmentación por umbral banda
        resultado = np.zeros_like(gray)
        resultado[(gray >= T1) & (gray <= T2)] = 255

        self.binary_image = resultado
        self.update_processed_image(resultado, f"Segmentación - Umbral Banda [{T1}, {T2}]")

    # ==================== NUEVAS FUNCIONES DE AJUSTE DE BRILLO ====================

    def equalizacion_uniforme(self):
        """Aplica equalización uniforme del histograma"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        resultado = cv2.equalizeHist(gray)
        self.update_processed_image(resultado, "Equalización Uniforme")

    def equalizacion_exponencial(self):
        """Aplica equalización exponencial"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Normalizar y aplicar transformación exponencial
        gray_norm = gray.astype(np.float32) / 255.0
        resultado = np.uint8(255 * (1 - np.exp(-gray_norm)))

        self.update_processed_image(resultado, "Equalización Exponencial")

    def equalizacion_rayleigh(self):
        """Aplica equalización Rayleigh"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Normalizar y aplicar transformación Rayleigh
        gray_norm = gray.astype(np.float32) / 255.0
        resultado = np.uint8(255 * np.sqrt(gray_norm))

        self.update_processed_image(resultado, "Equalización Rayleigh")

    def equalizacion_hipercubica(self):
        """Aplica equalización hipercúbica"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Normalizar y aplicar transformación hipercúbica
        gray_norm = gray.astype(np.float32) / 255.0
        resultado = np.uint8(255 * (gray_norm ** 4))

        self.update_processed_image(resultado, "Equalización Hipercúbica")

    def equalizacion_logaritmica(self):
        """Aplica equalización logarítmica hiperbólica"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Normalizar y aplicar transformación logarítmica
        gray_norm = gray.astype(np.float32) / 255.0
        resultado = np.uint8(255 * np.log1p(gray_norm) / np.log1p(1))

        self.update_processed_image(resultado, "Equalización Logarítmica")

    def funcion_potencia(self):
        """Aplica función potencia para ajuste de contraste"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Pedir exponente al usuario
        exponente = self.get_numeric_input("Función Potencia", "Exponente (ej: 2 para cuadrática):", 2.0)
        if exponente is None:
            return

        # Normalizar y aplicar función potencia
        gray_norm = gray.astype(np.float32) / 255.0
        resultado = np.uint8(255 * (gray_norm ** exponente))

        self.update_processed_image(resultado, f"Función Potencia (exponente: {exponente})")

    def correccion_gamma(self):
        """Aplica corrección gamma"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Pedir valor gamma al usuario
        gamma = self.get_numeric_input("Corrección Gamma", "Valor gamma (γ < 1 aclara, γ > 1 oscurece):", 1.0)
        if gamma is None:
            return

        # Aplicar corrección gamma
        resultado = np.power(gray / 255.0, gamma) * 255
        resultado = np.uint8(resultado)

        self.update_processed_image(resultado, f"Corrección Gamma (γ={gamma})")

    def desplazamiento_histograma(self):
        """Aplica desplazamiento del histograma para ajuste de brillo"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Pedir valor de desplazamiento
        desplazamiento = self.get_numeric_input("Desplazamiento Histograma",
                                                "Desplazamiento (+ aumenta brillo, - disminuye brillo):", 0)
        if desplazamiento is None:
            return

        # Aplicar desplazamiento
        resultado = cv2.add(gray, int(desplazamiento))
        resultado = np.clip(resultado, 0, 255).astype(np.uint8)

        self.update_processed_image(resultado, f"Desplazamiento Histograma ({desplazamiento})")

    def expansion_histograma(self):
        """Aplica expansión del histograma para aumentar contraste"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Calcular min y max actuales
        min_val = np.min(gray)
        max_val = np.max(gray)

        # Aplicar expansión lineal
        if max_val > min_val:
            resultado = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            resultado = gray

        self.update_processed_image(resultado, "Expansión de Histograma")

    def contraccion_histograma(self):
        """Aplica contracción del histograma para reducir contraste"""
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        # Pedir nuevo rango
        nuevo_min = simpledialog.askinteger("Contracción Histograma", "Nuevo valor mínimo:", initialvalue=50)
        nuevo_max = simpledialog.askinteger("Contracción Histograma", "Nuevo valor máximo:", initialvalue=200)

        if nuevo_min is None or nuevo_max is None:
            return

        # Aplicar contracción lineal
        resultado = ((gray - 0) / (255 - 0) * (nuevo_max - nuevo_min) + nuevo_min).astype(np.uint8)
        resultado = np.clip(resultado, nuevo_min, nuevo_max)

        self.update_processed_image(resultado, f"Contracción de Histograma [{nuevo_min}, {nuevo_max}]")

    # ==================== FUNCIONES DE CONTROL ====================

    def save_processing_state(self, operation_name):
        """Guarda el estado actual en el historial"""
        if self.processed_image is not None:
            self.processing_history.append({
                'name': operation_name,
                'image': self.processed_image.copy(),
                'gray': self.gray_image.copy() if self.gray_image is not None else None,
                'binary': self.binary_image.copy() if self.binary_image is not None else None
            })
            # Mantener solo los últimos 10 estados para no usar mucha memoria
            if len(self.processing_history) > 10:
                self.processing_history.pop(0)

    def undo_last_processing(self):
        """Deshace el último procesamiento"""
        if self.processing_history:
            # Remover el estado actual
            current_state = self.processing_history.pop()

            if self.processing_history:
                # Cargar el estado anterior
                previous_state = self.processing_history[-1]
                self.processed_image = previous_state['image'].copy()
                self.gray_image = previous_state['gray'].copy() if previous_state['gray'] is not None else None
                self.binary_image = previous_state['binary'].copy() if previous_state['binary'] is not None else None
            else:
                # Si no hay más historial, volver al original
                self.processed_image = self.original_image.copy() if self.original_image is not None else None
                self.gray_image = None
                self.binary_image = None

            self.display_image(self.processed_image, self.processed_canvas)
            self.show_histogram_auto()
            self.set_message(f"Deshecho: {current_state['name']}")
        else:
            messagebox.showinfo("Información", "No hay operaciones para deshacer")

    def show_processing_history(self):
        """Muestra el historial de procesamiento"""
        if not self.processing_history:
            messagebox.showinfo("Historial", "No hay operaciones en el historial")
            return

        history_text = "Historial de Procesamiento:\n\n"
        for i, state in enumerate(self.processing_history, 1):
            history_text += f"{i}. {state['name']}\n"

        messagebox.showinfo("Historial de Procesamiento", history_text)

    # ==================== FUNCIONES DE RUIDO ====================

    def add_salt_pepper_noise(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Función para agregar ruido sal y pimienta
        def agregar_ruido_sal_pimienta(imagen, cantidad=0.05):
            salida = np.copy(imagen)
            num_pixeles = int(cantidad * imagen.size)
            # Añadir ruido sal
            coords = [np.random.randint(0, i - 1, num_pixeles) for i in imagen.shape]
            salida[coords[0], coords[1]] = 255
            # Añadir ruido pimienta
            coords = [np.random.randint(0, i - 1, num_pixeles) for i in imagen.shape]
            salida[coords[0], coords[1]] = 0
            return salida

        cantidad = self.get_numeric_input("Ruido Sal y Pimienta", "Cantidad de ruido (0.01-0.2):", 0.05)
        if cantidad is None:
            return

        resultado = agregar_ruido_sal_pimienta(working_image, cantidad)
        self.update_processed_image(resultado, f"Ruido Sal y Pimienta (cantidad={cantidad})")

    def add_gaussian_noise(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Función para agregar ruido gaussiano
        def agregar_ruido_gaussiano(imagen, media=0, sigma=25):
            gauss = np.random.normal(media, sigma, imagen.shape).astype(np.int16)
            imagen_ruido = imagen.astype(np.int16) + gauss
            imagen_ruido = np.clip(imagen_ruido, 0, 255).astype(np.uint8)
            return imagen_ruido

        sigma = self.get_numeric_input("Ruido Gaussiano", "Desviación estándar (sigma):", 25)
        if sigma is None:
            return

        resultado = agregar_ruido_gaussiano(working_image, 0, sigma)
        self.update_processed_image(resultado, f"Ruido Gaussiano (sigma={sigma})")

    # ==================== FILTROS PASO BAJAS ====================

    def apply_average_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        kernel_size = simpledialog.askinteger("Filtro Promediador", "Tamaño del kernel (3, 5, 7, ...):", initialvalue=5)
        if kernel_size is None or kernel_size % 2 == 0:
            messagebox.showwarning("Advertencia", "El tamaño del kernel debe ser un número impar.")
            return

        resultado = cv2.blur(working_image, (kernel_size, kernel_size))
        self.update_processed_image(resultado, f"Filtro Promediador (kernel {kernel_size}x{kernel_size})")

    def apply_weighted_average_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Crear kernel personalizado con mayor peso en el centro
        kernel = np.array([[1, 1, 1],
                           [1, 5, 1],
                           [1, 1, 1]]) / 13

        resultado = cv2.filter2D(working_image, -1, kernel)
        self.update_processed_image(resultado, "Filtro Promediador Pesado")

    def apply_gaussian_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        kernel_size = simpledialog.askinteger("Filtro Gaussiano", "Tamaño del kernel (3, 5, 7, ...):", initialvalue=5)
        if kernel_size is None or kernel_size % 2 == 0:
            messagebox.showwarning("Advertencia", "El tamaño del kernel debe ser un número impar.")
            return

        sigma = self.get_numeric_input("Filtro Gaussiano", "Valor sigma:", 1.0)
        if sigma is None:
            return

        resultado = cv2.GaussianBlur(working_image, (kernel_size, kernel_size), sigma)
        self.update_processed_image(resultado, f"Filtro Gaussiano (kernel {kernel_size}x{kernel_size}, sigma={sigma})")

    def apply_bilateral_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        d = simpledialog.askinteger("Filtro Bilateral", "Diámetro de píxeles:", initialvalue=9)
        sigma_color = self.get_numeric_input("Filtro Bilateral", "Sigma Color:", 75.0)
        sigma_space = self.get_numeric_input("Filtro Bilateral", "Sigma Space:", 75.0)

        if d is None or sigma_color is None or sigma_space is None:
            return

        resultado = cv2.bilateralFilter(working_image, d, sigma_color, sigma_space)
        self.update_processed_image(resultado, f"Filtro Bilateral (d={d}, sigmaColor={sigma_color}, sigmaSpace={sigma_space})")

    # ==================== FILTROS NO LINEALES ====================

    def apply_median_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        kernel_size = simpledialog.askinteger("Filtro Mediana", "Tamaño del kernel (3, 5, 7, ...):", initialvalue=5)
        if kernel_size is None or kernel_size % 2 == 0:
            messagebox.showwarning("Advertencia", "El tamaño del kernel debe ser un número impar.")
            return

        resultado = cv2.medianBlur(working_image, kernel_size)
        self.update_processed_image(resultado, f"Filtro Mediana (kernel {kernel_size}x{kernel_size})")

    def apply_mode_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        def filtro_moda_counter(imagen, kernel_size=3):
            salida = np.copy(imagen)
            h, w = imagen.shape
            pad = kernel_size // 2
            imagen_padded = np.pad(imagen, pad, mode='constant', constant_values=0)

            for i in range(h):
                for j in range(w):
                    window = imagen_padded[i:i + kernel_size, j:j + kernel_size].flatten()

                    # Calcular la moda usando Counter
                    counter = Counter(window)
                    moda = counter.most_common(1)[0][0]

                    salida[i, j] = moda

            return salida.astype(np.uint8)

        kernel_size = simpledialog.askinteger("Filtro Moda", "Tamaño del kernel (3, 5, 7, ...):", initialvalue=3)
        if kernel_size is None or kernel_size % 2 == 0:
            messagebox.showwarning("Advertencia", "El tamaño del kernel debe ser un número impar.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image

        resultado = filtro_moda_counter(gray, kernel_size)
        self.update_processed_image(resultado, f"Filtro Moda (kernel {kernel_size}x{kernel_size})")

    def apply_max_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        kernel_size = simpledialog.askinteger("Filtro Máximo", "Tamaño del kernel (3, 5, 7, ...):", initialvalue=3)
        if kernel_size is None or kernel_size % 2 == 0:
            messagebox.showwarning("Advertencia", "El tamaño del kernel debe ser un número impar.")
            return

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        resultado = cv2.dilate(working_image, kernel)
        self.update_processed_image(resultado, f"Filtro Máximo (kernel {kernel_size}x{kernel_size})")

    def apply_min_filter(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        kernel_size = simpledialog.askinteger("Filtro Mínimo", "Tamaño del kernel (3, 5, 7, ...):", initialvalue=3)
        if kernel_size is None or kernel_size % 2 == 0:
            messagebox.showwarning("Advertencia", "El tamaño del kernel debe ser un número impar.")
            return

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        resultado = cv2.erode(working_image, kernel)
        self.update_processed_image(resultado, f"Filtro Mínimo (kernel {kernel_size}x{kernel_size})")

    # ==================== FILTROS PASO ALTAS ====================

    def apply_sobel(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            image = working_image

        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        bordes_sobel = cv2.magnitude(sobel_x, sobel_y)
        resultado = np.uint8(np.clip(bordes_sobel, 0, 255))
        self.update_processed_image(resultado, "Filtro Sobel")

    def apply_prewitt(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            image = working_image

        # Definir kernels de Prewitt
        kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

        # Aplicar filtros
        bordes_prewitt_x = cv2.filter2D(image, -1, kernel_prewitt_x)
        bordes_prewitt_y = cv2.filter2D(image, -1, kernel_prewitt_y)
        bordes_prewitt = cv2.addWeighted(bordes_prewitt_x, 0.5, bordes_prewitt_y, 0.5, 0)

        self.update_processed_image(bordes_prewitt, "Filtro Prewitt")

    def apply_roberts(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            image = working_image

        # Definir kernels de Roberts
        kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        # Aplicar filtros
        bordes_roberts_x = cv2.filter2D(image, -1, kernel_roberts_x)
        bordes_roberts_y = cv2.filter2D(image, -1, kernel_roberts_y)
        bordes_roberts = cv2.addWeighted(bordes_roberts_x, 0.5, bordes_roberts_y, 0.5, 0)

        self.update_processed_image(bordes_roberts, "Filtro Roberts")

    def apply_canny(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            image = working_image

        threshold1 = self.get_numeric_input("Canny", "Umbral inferior:", 100)
        threshold2 = self.get_numeric_input("Canny", "Umbral superior:", 200)

        if threshold1 is None or threshold2 is None:
            return

        resultado = cv2.Canny(image, threshold1, threshold2)
        self.update_processed_image(resultado, f"Filtro Canny (umbrales: {threshold1}, {threshold2})")

    def apply_kirsch(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            image = working_image

        # Definir kernels de Kirsch (algunos ejemplos)
        kirsch_kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),  # Norte
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),  # Noreste
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),  # Este
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),  # Sureste
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),  # Sur
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),  # Suroeste
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),  # Oeste
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32)  # Noroeste
        ]

        # Aplicar cada kernel y tomar el máximo
        responses = [cv2.filter2D(image, -1, kernel) for kernel in kirsch_kernels]
        bordes_kirsch = np.max(responses, axis=0)

        resultado = np.uint8(np.clip(bordes_kirsch, 0, 255))
        self.update_processed_image(resultado, "Filtro Kirsch")

    def apply_laplacian(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            image = working_image

        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        resultado = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        self.update_processed_image(resultado, "Filtro Laplaciano")

    # ==================== MÁSCARAS DIRECCIONALES ====================

    def apply_directional_filter(self, direction):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Convertir a escala de grises si es necesario
        if len(working_image.shape) == 3:
            image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            image = working_image

        # Definir kernels según la dirección
        if direction == 'horizontal':
            kernel = np.array([[-1, -1, -1],
                               [2, 2, 2],
                               [-1, -1, -1]], dtype=np.float32)
            title = "Líneas Horizontales"
        elif direction == 'vertical':
            kernel = np.array([[-1, 2, -1],
                               [-1, 2, -1],
                               [-1, 2, -1]], dtype=np.float32)
            title = "Líneas Verticales"
        elif direction == 'diagonal_main':
            kernel = np.array([[2, -1, -1],
                               [-1, 2, -1],
                               [-1, -1, 2]], dtype=np.float32)
            title = "Diagonal Principal"
        elif direction == 'diagonal_secondary':
            kernel = np.array([[-1, -1, 2],
                               [-1, 2, -1],
                               [2, -1, -1]], dtype=np.float32)
            title = "Diagonal Secundaria"
        else:
            return

        # Aplicar filtro
        filtered = cv2.filter2D(image, -1, kernel)
        resultado = np.uint8(np.clip(filtered, 0, 255))
        self.update_processed_image(resultado, f"Máscara {title}")

    # ==================== PREPROCESAMIENTO ====================

    def convert_to_grayscale(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        if len(working_image.shape) == 3:
            self.gray_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            self.update_processed_image(self.gray_image, "Escala de Grises")
        else:
            messagebox.showinfo("Información", "La imagen ya está en escala de grises.")

    def threshold_image(self):
        self.apply_threshold(127)

    def custom_threshold(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        threshold_value = self.get_numeric_input("Umbralización", "Ingresa el valor de umbral (0-255):", 127)
        if threshold_value is None:
            return

        threshold_value = max(0, min(255, threshold_value))
        self.apply_threshold(threshold_value)

    def apply_threshold(self, threshold_value):
        working_image = self.get_working_image()
        if working_image is None:
            return

        # Siempre convertir a escala de grises para la umbralización
        if len(working_image.shape) == 3:
            # Si es color, convertir a grises
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            # Si ya es grises, usar directamente
            gray = working_image

        # Actualizar gray_image con la imagen actual
        self.gray_image = gray.copy()

        # Aplicar umbral
        _, binary_result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Actualizar ambos estados
        self.binary_image = binary_result
        self.update_processed_image(binary_result, f"Umbralización (umbral={threshold_value})")

    def connected_components(self, connectivity):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # Usar siempre la imagen binaria actual si existe
        if self.binary_image is not None:
            # Usar la imagen binaria existente
            binary_for_components = self.binary_image
        else:
            # Si no hay imagen binaria, usar la imagen procesada actual
            # Convertir a escala de grises si es necesario
            if len(working_image.shape) == 3:
                working_gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            else:
                working_gray = working_image

            # Aplicar umbral automático
            _, binary_for_components = cv2.threshold(working_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, labels = cv2.connectedComponents(binary_for_components, connectivity=connectivity)
        self.set_message(f"Vecindad {connectivity}: {num_labels - 1} objetos detectados.")
        labels_normalized = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        resultado = cv2.applyColorMap(labels_normalized, cv2.COLORMAP_JET)
        self.update_processed_image(resultado, f"Componentes Conexas Vecindad {connectivity}")

    def contour_objects(self):
        working_image = self.get_working_image()
        if working_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        #  Si ya hay una imagen binaria, usarla directamente
        if self.binary_image is not None:
            # Usar la imagen binaria existente
            binary_for_contours = self.binary_image
        else:
            # Si no hay imagen binaria, crear una desde la imagen procesada actual
            # Convertir a escala de grises si es necesario
            if len(working_image.shape) == 3:
                working_gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            else:
                working_gray = working_image

            # Aplicar umbral automático
            _, binary_for_contours = cv2.threshold(working_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.binary_image = binary_for_contours
            self.gray_image = working_gray

        contours, _ = cv2.findContours(binary_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar contornos sobre la imagen base apropiada
        if working_image is not None and working_image is not self.binary_image:
            # Usar la imagen procesada actual como base
            if len(working_image.shape) == 3:
                result_image = working_image.copy()
            else:
                result_image = cv2.cvtColor(working_image, cv2.COLOR_GRAY2BGR)
        else:
            # Usar la imagen original como base
            if len(self.original_image.shape) == 3:
                result_image = self.original_image.copy()
            else:
                result_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)

        for i, contour in enumerate(contours):
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(result_image, f'{i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.update_processed_image(result_image, f"Contornear Objetos ({len(contours)} objetos)")

   

    def set_message(self, text):
        if self.message_label is not None:
            self.message_label.config(text=text)
        print(text)

    def save_processed_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar.")
            return
        file_path = filedialog.asksaveasfilename(title="Guardar imagen procesada", defaultextension=".png",
                                                 filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"),
                                                            ("BMP", "*.bmp"), ("TIFF", "*.tiff")])
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
            # Guardar estado inicial en el historial
            self.processing_history = [{
                'name': 'Imagen Original',
                'image': self.processed_image.copy(),
                'gray': None,
                'binary': None
            }]
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
            self.processing_history = []
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
        self.processing_history = []
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
        self.processing_history = [{
            'name': 'Imagen Original',
            'image': self.processed_image.copy(),
            'gray': None,
            'binary': None
        }]
        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()
        self.set_message("Imagen procesada restablecida al estado original.")

    def display_image(self, image, canvas):
        canvas.delete("all")
        if image is None:
            canvas.create_text(canvas.winfo_reqwidth() // 2, canvas.winfo_reqheight() // 2,
                               text="No hay imagen", fill="black", font=('Arial', 12))
            return
        try:
            # Si la imagen es en escala de grises, mantenerla así
            if image.ndim == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
            canvas.create_text(200, 120, text=f"Error: {e}", fill='red')

    # ----------------- operaciones aritméticas -----------------
    def arithmetic_operation_scalar(self, operation):
        src = self.get_working_image()
        if src is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        # si la imagen es binaria: permitir entrada pero no cambiar visualmente
        if self.binary_image is not None and self.processed_image is not None and np.array_equal(self.processed_image,
                                                                                                 self.binary_image):
            if operation == 'add':
                value = self.get_numeric_input("Suma con Escalar", "Ingresa el valor a sumar a cada píxel:", 50)
                if value is None: return
                self.set_message(
                    f"Suma con escalar +{value} solicitada sobre imagen binarizada (sin cambios visibles).")
            elif operation == 'subtract':
                value = self.get_numeric_input("Resta con Escalar", "Ingresa el valor a restar a cada píxel:", 50)
                if value is None: return
                self.set_message(
                    f"Resta con escalar -{value} solicitada sobre imagen binarizada (sin cambios visibles).")
            elif operation == 'multiply':
                value = self.get_numeric_input("Multiplicación con Escalar", "Ingresa el factor de multiplicación:",
                                               1.2)
                if value is None: return
                self.set_message(
                    f"Multiplicación con escalar x{value} solicitada sobre imagen binarizada (sin cambios visibles).")
            self.display_image(self.processed_image, self.processed_canvas)
            self.show_histogram_auto()
            return

        # pedir valor y aplicar sobre imagen original
        if operation == 'add':
            value = self.get_numeric_input("Suma con Escalar", "Ingresa el valor a sumar a cada píxel:", 50)
            if value is None: return
            resultado = cv2.add(src, value)
            self.update_processed_image(resultado, f"Suma con escalar +{value}")
        elif operation == 'subtract':
            value = self.get_numeric_input("Resta con Escalar", "Ingresa el valor a restar a cada píxel:", 50)
            if value is None: return
            resultado = cv2.subtract(src, value)
            self.update_processed_image(resultado, f"Resta con escalar -{value}")
        elif operation == 'multiply':
            value = self.get_numeric_input("Multiplicación con Escalar", "Ingresa el factor de multiplicación:", 1.2)
            if value is None: return
            res = cv2.multiply(src.astype(np.float32), float(value))
            resultado = np.clip(res, 0, 255).astype(np.uint8)
            self.update_processed_image(resultado, f"Multiplicación con escalar x{value}")

    def arithmetic_operation_images(self, operation):
        imgA = self.get_working_image()
        if imgA is None:
            messagebox.showwarning("Advertencia", "Primero carga la imagen 1.")
            return
        if self.image2 is None:
            messagebox.showwarning("Advertencia", "Para esta operación necesitas cargar la imagen 2.")
            return

        # si binaria
        if self.binary_image is not None and self.processed_image is not None and np.array_equal(self.processed_image,
                                                                                                 self.binary_image):
            self.set_message(
                "Operación aritmética entre imágenes solicitada sobre imagen binarizada (sin cambios visibles).")
            self.display_image(self.processed_image, self.processed_canvas)
            self.show_histogram_auto()
            return

        h1, w1 = imgA.shape[:2]
        h2, w2 = self.image2.shape[:2]
        h, w = min(h1, h2), min(w1, w2)
        A = cv2.resize(imgA, (w, h))
        B = cv2.resize(self.image2, (w, h))

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
                result = cv2.add(A_proc, B_proc);
                operation_name = "Suma de imágenes"
            elif operation == 'subtract':
                result = cv2.subtract(A_proc, B_proc);
                operation_name = "Resta de imágenes"
            elif operation == 'multiply':
                result = cv2.multiply(A_proc, B_proc);
                operation_name = "Multiplicación de imágenes"
            else:
                messagebox.showerror("Error", "Operación desconocida.")
                return
            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            self.update_processed_image(result, operation_name)
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
            h1, w1 = self.original_image.shape[:2]
            h2, w2 = self.image2.shape[:2]
            h, w = min(h1, h2), min(w1, w2)
            A = cv2.resize(self.original_image, (w, h))
            B = cv2.resize(self.image2, (w, h))
            if operation == 'and':
                res = cv2.bitwise_and(A, B);
                operation_name = "Operación: AND"
            elif operation == 'or':
                res = cv2.bitwise_or(A, B);
                operation_name = "Operación: OR"
            elif operation == 'xor':
                res = cv2.bitwise_xor(A, B);
                operation_name = "Operación: XOR"
            else:
                messagebox.showerror("Error", "Operación desconocida.")
                return
            self.update_processed_image(res, operation_name)
        else:
            resultado = cv2.bitwise_not(self.original_image)
            self.update_processed_image(resultado, "Operación: NOT")

    # ----------------- histogramas -----------------
    def clear_histogram(self):
        self.hist_ax_orig.clear()
        self.hist_ax_proc.clear()
        self.hist_ax_orig.set_title('Original')
        self.hist_ax_proc.set_title('Procesada')
        self.hist_canvas.draw()

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

        self.hist_ax_orig.clear();
        self.hist_ax_proc.clear()

        # original
        orig = self.original_image
        if mode_orig == 'rgb':
            colors = ('b', 'g', 'r');
            names = ('Azul', 'Verde', 'Rojo')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([orig], [i], None, [256], [0, 256])
                self.hist_ax_orig.plot(hist, label=names[i])
            self.hist_ax_orig.set_title('Original - RGB');
            self.hist_ax_orig.legend(fontsize='small')
        else:
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if orig.ndim == 3 else orig
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            self.hist_ax_orig.plot(hist, label='Grises')
            self.hist_ax_orig.set_title('Original - Grises');
            self.hist_ax_orig.legend(fontsize='small')

        # procesada
        proc = self.processed_image
        if proc is None:
            self.hist_ax_proc.set_title('Procesada - (vacía)')
        else:
            if mode_proc == 'binary':
                proc_gray = proc if proc.ndim == 2 else cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                hist_p = cv2.calcHist([proc_gray], [0], None, [256], [0, 256])
                self.hist_ax_proc.plot(hist_p, label='Binarizada');
                self.hist_ax_proc.set_title('Procesada - Binarizada')
                self.hist_ax_proc.legend(fontsize='small')
            elif mode_proc == 'rgb':
                colors = ('b', 'g', 'r');
                names = ('Azul', 'Verde', 'Rojo')
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([proc], [i], None, [256], [0, 256])
                    self.hist_ax_proc.plot(hist, label=names[i])
                self.hist_ax_proc.set_title('Procesada - RGB');
                self.hist_ax_proc.legend(fontsize='small')
            else:
                pgray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY) if proc.ndim == 3 else proc
                hist_p = cv2.calcHist([pgray], [0], None, [256], [0, 256])
                self.hist_ax_proc.plot(hist_p, label='Grises');
                self.hist_ax_proc.set_title('Procesada - Grises')
                self.hist_ax_proc.legend(fontsize='small')

        self.hist_ax_orig.set_xlabel('Valor de píxel');
        self.hist_ax_orig.set_ylabel('Frecuencia')
        self.hist_ax_proc.set_xlabel('Valor de píxel');
        self.hist_ax_proc.set_ylabel('Frecuencia')
        self.hist_ax_orig.grid(True, alpha=0.3);
        self.hist_ax_proc.grid(True, alpha=0.3)
        self.hist_canvas.draw()
        self.set_message("Histogramas actualizados.")

    def update_display(self):
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas)
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_canvas)

    # ==================== FUNCIONES DE REMOVER FILTROS ====================

    def remove_noise_filters(self):
        """Quita solo filtros de ruido, mantiene el resto del procesamiento"""
        if self.processing_history:
            # Buscar el último estado que no tenga ruido
            for i in range(len(self.processing_history) - 1, -1, -1):
                state = self.processing_history[i]
                if 'ruido' not in state['name'].lower():
                    self.processed_image = state['image'].copy()
                    self.gray_image = state['gray'].copy() if state['gray'] is not None else None
                    self.binary_image = state['binary'].copy() if state['binary'] is not None else None

                    # Actualizar el historial con este estado
                    self.processing_history = self.processing_history[:i + 1]
                    break
            else:
                # Si no encuentra, volver al original
                self.processed_image = self.original_image.copy() if self.original_image is not None else None
                self.gray_image = None
                self.binary_image = None
                self.processing_history = [{
                    'name': 'Imagen Original',
                    'image': self.processed_image.copy() if self.processed_image is not None else None,
                    'gray': None,
                    'binary': None
                }]

        self.display_image(self.processed_image, self.processed_canvas)
        self.show_histogram_auto()
        self.set_message("Filtros de ruido removidos")

    def remove_smoothing_filters(self):
        """Quita solo filtros de suavizado, mantiene el resto"""
        pass

    def remove_nonlinear_filters(self):
        """Quita solo filtros no lineales"""
        pass

    def remove_edge_filters(self):
        """Quita solo filtros de detección de bordes"""
        pass

    def remove_directional_filters(self):
        """Quita solo máscaras direccionales"""
        pass

    def remove_arithmetic_operations(self):
        """Quita solo operaciones aritméticas con escalar"""
        pass

    def remove_image_operations(self):
        """Quita solo operaciones entre imágenes"""
        pass

    def remove_logical_operations(self):
        """Quita solo operaciones lógicas"""
        pass

    def remove_preprocessing(self):
        """Quita solo preprocesamiento básico (vuelve a color)"""
        pass

    def remove_labeling(self):
        """Quita solo etiquetado de componentes"""
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.after(120, app.update_display)
    root.mainloop()
