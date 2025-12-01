import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# PyQt6 imports
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QListWidgetItem,
                             QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QFrame, QStackedWidget, QCheckBox, QGridLayout,
                             QRadioButton, QMessageBox, QScrollArea,
                             QSizePolicy, QGroupBox, QAbstractItemView,
                             QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap

# Matplotlib imports - ВАЖНО: сначала установить backend, потом импортировать остальное
import matplotlib

matplotlib.use('QtAgg')  # Используем QtAgg для совместимости с PyQt6

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import cm

# Импортируем UI
try:
    from form.visualization import Ui_Visualization
except ImportError:
    print("Ошибка: Не удалось импортировать Ui_Visualization из form.visualization")
    print("Создаю простой UI для теста...")


    # Создаем простой заглушечный класс UI для теста
    class Ui_Visualization:
        def setupUi(self, Visualization):
            Visualization.setObjectName("Visualization")
            Visualization.resize(800, 600)

            self.centralwidget = QWidget(Visualization)
            self.verticalLayout = QVBoxLayout(self.centralwidget)

            self.label = QLabel("Тестовый интерфейс", self.centralwidget)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.verticalLayout.addWidget(self.label)

            self.btn_test = QPushButton("Тест", self.centralwidget)
            self.verticalLayout.addWidget(self.btn_test)

            Visualization.setCentralWidget(self.centralwidget)

        def retranslateUi(self, Visualization):
            pass


class ChartConfig:
    """Класс для хранения конфигурации графика для matplotlib/pandas"""

    def __init__(self, chart_type: str, title: str, data_config: Dict, styling: Dict,
                 fig_settings: Dict = None, created_at: str = None, id: str = None):
        self.chart_type = chart_type
        self.title = title
        self.data_config = data_config
        self.styling = styling
        self.fig_settings = fig_settings or {
            'figsize_width': 10,
            'figsize_height': 6,
            'dpi': 100
        }
        self.created_at = created_at or datetime.now().isoformat()
        self.id = id or f"chart_{int(datetime.now().timestamp())}_{hash(title)}"

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'chart_type': self.chart_type,
            'title': self.title,
            'data_config': self.data_config,
            'styling': self.styling,
            'fig_settings': self.fig_settings,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChartConfig':
        return cls(
            id=data.get('id'),
            chart_type=data['chart_type'],
            title=data['title'],
            data_config=data.get('data_config', {}),
            styling=data.get('styling', {}),
            fig_settings=data.get('fig_settings', {
                'figsize_width': 10,
                'figsize_height': 6,
                'dpi': 100
            }),
            created_at=data.get('created_at', datetime.now().isoformat())
        )


class ChartManager:
    """Менеджер для работы с конфигурациями графиков"""

    def __init__(self, app_data_dir: str = "data/app_data"):
        self.app_data_dir = app_data_dir
        os.makedirs(app_data_dir, exist_ok=True)

    def get_charts_file_path(self, dataset_name: str) -> str:
        base_name = os.path.splitext(dataset_name)[0]
        return os.path.join(self.app_data_dir, f"{base_name}_charts.json")

    def save_charts(self, dataset_name: str, charts: List[ChartConfig]):
        file_path = self.get_charts_file_path(dataset_name)
        charts_data = [chart.to_dict() for chart in charts]

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(charts_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Ошибка при сохранении графиков: {e}")
            return False

    def load_charts(self, dataset_name: str) -> List[ChartConfig]:
        file_path = self.get_charts_file_path(dataset_name)

        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                charts_data = json.load(f)

            return [ChartConfig.from_dict(data) for data in charts_data]
        except Exception as e:
            print(f"Ошибка при загрузке графиков: {e}")
            return []


class MplCanvas(FigureCanvas):
    """Кастомный виджет для отображения matplotlib графиков"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class ChartWidget(QWidget):
    """Виджет для отображения одного графика с тулбаром"""

    def __init__(self, chart_config: ChartConfig, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.chart_config = chart_config
        self.data = data
        self.canvas = None
        self.toolbar = None
        self.init_ui()
        self.plot_chart()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)

        title_label = QLabel(f"<b>{self.chart_config.title}</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #1e3a5f; font-size: 14px; margin: 5px;")
        layout.addWidget(title_label)

        type_label = QLabel(f"Тип: {self.chart_config.chart_type}")
        type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        type_label.setStyleSheet("color: #4a5568; font-size: 12px; margin: 2px;")
        layout.addWidget(type_label)

        fig_settings = self.chart_config.fig_settings
        width = fig_settings.get('figsize_width', 5)
        height = fig_settings.get('figsize_height', 4)
        dpi = fig_settings.get('dpi', 100)

        self.canvas = MplCanvas(self, width=width, height=height, dpi=dpi)
        layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.setLayout(layout)

    def plot_chart(self):
        try:
            self.canvas.axes.clear()

            chart_type = self.chart_config.chart_type
            data_config = self.chart_config.data_config
            styling = self.chart_config.styling

            # Сохраняем текущие настройки для восстановления
            original_rcParams = plt.rcParams.copy()

            try:
                if chart_type == "Линейный график (plot)":
                    self._plot_line_chart(data_config, styling)
                elif chart_type == "Столбчатая диаграмма (bar)":
                    self._plot_bar_chart(data_config, styling)
                elif chart_type == "Круговая диаграмма (pie)":
                    self._plot_pie_chart(data_config, styling)
                elif chart_type == "Гистограмма (hist)":
                    self._plot_histogram(data_config, styling)
                elif chart_type == "Диаграмма рассеяния (scatter)":
                    self._plot_scatter(data_config, styling)
                elif chart_type == "Box Plot (boxplot)":
                    self._plot_boxplot(data_config, styling)
                elif chart_type == "Площадной график (area)":
                    self._plot_area_chart(data_config, styling)
                elif chart_type == "График плотности (kde)":
                    self._plot_kde_chart(data_config, styling)

                if styling.get('show_grid', True):
                    self.canvas.axes.grid(True, alpha=0.3, linestyle='--')

                self.canvas.fig.tight_layout()
                self.canvas.draw()

            finally:
                # Восстанавливаем оригинальные настройки
                plt.rcParams.update(original_rcParams)

        except Exception as e:
            print(f"Ошибка при построении графика: {e}")
            import traceback
            traceback.print_exc()
            self.canvas.axes.text(0.5, 0.5, f"Ошибка: {str(e)[:50]}...",
                                  ha='center', va='center',
                                  transform=self.canvas.axes.transAxes,
                                  color='red', fontsize=10)
            self.canvas.draw()

    def _plot_line_chart(self, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_cols = data_config.get('y_columns', [])

        if not x_col or not y_cols:
            raise ValueError("Не указаны данные для графика")

        for i, y_col in enumerate(y_cols):
            if x_col in self.data.columns and y_col in self.data.columns:
                line_style = styling.get('line_style', 'solid (сплошная)').split(' ')[0]
                line_width = styling.get('line_width', 2.0)
                markers = styling.get('markers', 'None (нет)').split(' ')[0]
                markersize = styling.get('markersize', 6)
                color = styling.get('color', 'blue (синий)').split(' ')[0]
                alpha = styling.get('alpha', 1.0)

                x_data = self.data[x_col]
                y_data = self.data[y_col]

                # Очистка от NaN значений
                mask = x_data.notna() & y_data.notna()
                x_clean = x_data[mask]
                y_clean = y_data[mask]

                if len(x_clean) > 0:
                    self.canvas.axes.plot(
                        x_clean,
                        y_clean,
                        linestyle=line_style,
                        linewidth=line_width,
                        marker=markers if markers != 'None' else None,
                        markersize=markersize,
                        color=color,
                        alpha=alpha,
                        label=y_col
                    )

        xlabel = styling.get('xlabel', x_col)
        ylabel = styling.get('ylabel', 'Значения')

        self.canvas.axes.set_xlabel(xlabel)
        self.canvas.axes.set_ylabel(ylabel)

        if styling.get('show_legend', True) and y_cols:
            self.canvas.axes.legend()

        self.canvas.axes.set_title(self.chart_config.title)

    def _plot_bar_chart(self, data_config: Dict, styling: Dict):
        categories_col = data_config.get('categories_column')
        values_col = data_config.get('values_column')

        if not categories_col or not values_col:
            raise ValueError("Не указаны данные для графика")

        if categories_col in self.data.columns and values_col in self.data.columns:
            orientation = styling.get('orientation', 'vertical (вертикальная)').split(' ')[0]
            color = styling.get('color', 'steelblue').split(' ')[0]
            edgecolor = styling.get('edgecolor', 'black (черный)').split(' ')[0]
            edgewidth = styling.get('edgewidth', 1.0)
            alpha = styling.get('alpha', 0.8)
            width = styling.get('width', 0.8)

            categories = self.data[categories_col].astype(str)
            values = self.data[values_col]

            if orientation == 'vertical':
                bars = self.canvas.axes.bar(
                    categories,
                    values,
                    color=color,
                    edgecolor=edgecolor if edgecolor != 'none' else None,
                    linewidth=edgewidth,
                    alpha=alpha,
                    width=width
                )
                self.canvas.axes.set_xlabel(categories_col)
                self.canvas.axes.set_ylabel(values_col)
            else:
                bars = self.canvas.axes.barh(
                    categories,
                    values,
                    color=color,
                    edgecolor=edgecolor if edgecolor != 'none' else None,
                    linewidth=edgewidth,
                    alpha=alpha,
                    height=width
                )
                self.canvas.axes.set_xlabel(values_col)
                self.canvas.axes.set_ylabel(categories_col)

            plt.setp(self.canvas.axes.get_xticklabels(), rotation=45, ha='right')
            self.canvas.axes.set_title(self.chart_config.title)

    def _plot_pie_chart(self, data_config: Dict, styling: Dict):
        labels_col = data_config.get('labels_column')
        values_col = data_config.get('values_column')

        if not labels_col or not values_col:
            raise ValueError("Не указаны данные для графика")

        if labels_col in self.data.columns and values_col in self.data.columns:
            start_angle = styling.get('start_angle', 90)
            explode = styling.get('explode', 0.1)
            autopct = styling.get('autopct', 'Не показывать')
            show_shadow = styling.get('shadow', False)
            colormap = styling.get('colormap', 'tab20c')

            labels = self.data[labels_col].astype(str)
            values = self.data[values_col]

            # Подготовка параметров для pie chart
            autopct_format = None
            if autopct != 'Не показывать':
                autopct_format = autopct

            # Создание explode массива
            if explode > 0:
                explode_values = [explode] + [0] * (len(values) - 1)
            else:
                explode_values = None

            # Получение цветов из colormap - исправленная версия
            try:
                # Для matplotlib >= 3.7
                cmap = plt.colormaps[colormap]
            except:
                # Для старых версий matplotlib
                cmap = cm.get_cmap(colormap)
            colors = cmap(np.linspace(0, 1, len(values)))

            wedges, texts, autotexts = self.canvas.axes.pie(
                values,
                labels=labels,
                autopct=autopct_format,
                startangle=start_angle,
                explode=explode_values,
                shadow=show_shadow,
                colors=colors
            )

            # Настройка внешнего вида
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            self.canvas.axes.set_title(self.chart_config.title)
            self.canvas.axes.axis('equal')  # Equal aspect ratio для круговой диаграммы

    def _plot_histogram(self, data_config: Dict, styling: Dict):
        column = data_config.get('column')

        if not column:
            raise ValueError("Не указана колонка для гистограммы")

        if column in self.data.columns:
            bins = styling.get('bins', 'auto (автоматически)')
            hist_type = styling.get('hist_type', 'bar (столбчатая)').split(' ')[0]
            color = styling.get('color', 'steelblue')
            alpha = styling.get('alpha', 0.7)
            show_density = styling.get('density', False)
            show_cumulative = styling.get('cumulative', False)

            data = self.data[column].dropna()

            if bins.startswith('auto'):
                bins = 'auto'
            else:
                try:
                    bins = int(bins)
                except:
                    bins = 10

            self.canvas.axes.hist(
                data,
                bins=bins,
                histtype=hist_type,
                color=color,
                alpha=alpha,
                density=show_density,
                cumulative=show_cumulative,
                edgecolor='black'
            )

            self.canvas.axes.set_xlabel(column)
            self.canvas.axes.set_ylabel('Плотность' if show_density else 'Частота')
            self.canvas.axes.set_title(self.chart_config.title)

            if show_density:
                # Добавление линии плотности
                try:
                    from scipy import stats
                    if len(data) > 1:
                        kde = stats.gaussian_kde(data)
                        x_range = np.linspace(data.min(), data.max(), 100)
                        self.canvas.axes.plot(x_range, kde(x_range), 'r-', linewidth=2,
                                              label='Плотность')
                        self.canvas.axes.legend()
                except ImportError:
                    print("Для показа плотности установите scipy: pip install scipy")

    def _plot_scatter(self, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_col = data_config.get('y_column')
        color_col = data_config.get('color_column')

        if not x_col or not y_col:
            raise ValueError("Не указаны данные для графика")

        if x_col in self.data.columns and y_col in self.data.columns:
            point_size = styling.get('point_size', 50)
            point_alpha = styling.get('point_alpha', 0.6)
            marker = styling.get('marker', 'o (круг)').split(' ')[0]
            show_regression = styling.get('regression', False)
            colormap = styling.get('colormap', 'viridis')

            x_data = self.data[x_col]
            y_data = self.data[y_col]

            # Очистка от NaN значений
            mask = x_data.notna() & y_data.notna()
            x_clean = x_data[mask]
            y_clean = y_data[mask]

            if color_col and color_col in self.data.columns:
                color_data = self.data[color_col][mask]
                scatter = self.canvas.axes.scatter(
                    x_clean,
                    y_clean,
                    c=color_data,
                    s=point_size,
                    alpha=point_alpha,
                    marker=marker,
                    cmap=colormap
                )
                plt.colorbar(scatter, ax=self.canvas.axes, label=color_col)
            else:
                self.canvas.axes.scatter(
                    x_clean,
                    y_clean,
                    s=point_size,
                    alpha=point_alpha,
                    marker=marker,
                    color='blue'
                )

            if show_regression and len(x_clean) > 1:
                # Линейная регрессия
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                self.canvas.axes.plot(
                    x_clean,
                    p(x_clean),
                    "r--",
                    linewidth=2,
                    label='Линия тренда'
                )
                self.canvas.axes.legend()

            self.canvas.axes.set_xlabel(x_col)
            self.canvas.axes.set_ylabel(y_col)
            self.canvas.axes.set_title(self.chart_config.title)

    def _plot_boxplot(self, data_config: Dict, styling: Dict):
        category_col = data_config.get('category_column')
        values_col = data_config.get('values_column')

        if not values_col:
            raise ValueError("Не указана колонка значений")

        orientation = styling.get('orientation', 'vertical (вертикальная)').split(' ')[0]
        show_points = styling.get('show_points', 'outliers (только выбросы)').split(' ')[0]
        show_notch = styling.get('notch', False)
        color = styling.get('color', 'lightblue')
        linewidth = styling.get('linewidth', 1.5)
        whis = styling.get('whis', 1.5)

        if category_col and category_col in self.data.columns:
            # Группировка по категориям
            data = []
            labels = []

            for category in sorted(self.data[category_col].astype(str).unique()):
                subset = self.data[self.data[category_col].astype(str) == category][values_col]
                if len(subset) > 0:
                    data.append(subset.dropna().values)
                    labels.append(str(category))
        else:
            # Один боксплот для всей колонки
            data = [self.data[values_col].dropna().values]
            labels = [values_col]

        showfliers = show_points in ['outliers', 'all']
        if orientation == 'vertical':
            bp = self.canvas.axes.boxplot(
                data,
                labels=labels,
                notch=show_notch,
                showfliers=showfliers,
                patch_artist=True,
                boxprops=dict(facecolor=color, linewidth=linewidth),
                whiskerprops=dict(linewidth=linewidth),
                capprops=dict(linewidth=linewidth),
                medianprops=dict(linewidth=linewidth, color='red'),
                flierprops=dict(marker='o', markersize=5, alpha=0.5),
                whis=whis
            )
            self.canvas.axes.set_ylabel(values_col)
        else:
            bp = self.canvas.axes.boxplot(
                data,
                labels=labels,
                vert=False,
                notch=show_notch,
                showfliers=showfliers,
                patch_artist=True,
                boxprops=dict(facecolor=color, linewidth=linewidth),
                whiskerprops=dict(linewidth=linewidth),
                capprops=dict(linewidth=linewidth),
                medianprops=dict(linewidth=linewidth, color='red'),
                flierprops=dict(marker='o', markersize=5, alpha=0.5),
                whis=whis
            )
            self.canvas.axes.set_xlabel(values_col)

        plt.setp(self.canvas.axes.get_xticklabels(), rotation=45, ha='right')
        self.canvas.axes.set_title(self.chart_config.title)

    def _plot_area_chart(self, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_cols = data_config.get('y_columns', [])

        if not x_col or not y_cols:
            raise ValueError("Не указаны данные для графика")

        if x_col in self.data.columns:
            alpha = styling.get('alpha', 0.5)
            colormap = styling.get('colormap', 'viridis')

            x_data = self.data[x_col]

            # Собираем все y данные
            y_data_list = []
            valid_cols = []

            for y_col in y_cols:
                if y_col in self.data.columns:
                    y_data = self.data[y_col]
                    mask = x_data.notna() & y_data.notna()
                    if len(x_data[mask]) > 0:
                        y_data_list.append(y_data[mask].values)
                        valid_cols.append(y_col)

            if y_data_list:
                # Создание stacked area chart
                y_stack = np.vstack(y_data_list)

                # Исправленная версия получения colormap
                try:
                    cmap = plt.colormaps[colormap]
                except:
                    cmap = cm.get_cmap(colormap)
                colors = cmap(np.linspace(0, 1, len(y_data_list)))

                self.canvas.axes.stackplot(
                    x_data[x_data.notna()].values,
                    *y_data_list,
                    labels=valid_cols,
                    colors=colors,
                    alpha=alpha
                )

                self.canvas.axes.set_xlabel(x_col)
                self.canvas.axes.set_ylabel('Значения')
                self.canvas.axes.legend()
                self.canvas.axes.set_title(self.chart_config.title)

    def _plot_kde_chart(self, data_config: Dict, styling: Dict):
        columns = data_config.get('columns', [])

        if not columns:
            raise ValueError("Не указаны колонки для KDE")

        alpha = styling.get('alpha', 0.5)
        colormap = styling.get('colormap', 'viridis')
        show_fill = styling.get('fill', True)
        bandwidth = styling.get('bandwidth', None)

        try:
            from scipy import stats
        except ImportError:
            print("Для KDE установите scipy: pip install scipy")
            return

        # Исправленная версия получения colormap
        try:
            cmap = plt.colormaps[colormap]
        except:
            cmap = cm.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, len(columns)))

        for i, column in enumerate(columns):
            if column in self.data.columns:
                data = self.data[column].dropna()
                if len(data) > 1:
                    kde = stats.gaussian_kde(data, bw_method=bandwidth)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    y_kde = kde(x_range)

                    if show_fill:
                        self.canvas.axes.fill_between(x_range, y_kde, alpha=alpha * 0.7,
                                                      color=colors[i])

                    self.canvas.axes.plot(x_range, y_kde, color=colors[i], linewidth=2,
                                          label=column)

        self.canvas.axes.set_xlabel('Значения')
        self.canvas.axes.set_ylabel('Плотность вероятности')
        self.canvas.axes.legend()
        self.canvas.axes.set_title(self.chart_config.title)


class VisualizationWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, filename: str, parent=None):
        super().__init__(parent)

        self.filename = filename
        self.data = None
        self.charts = []
        self.current_chart_index = -1
        self.layout_config = {'rows': 1, 'cols': 1}
        self.chart_manager = ChartManager()

        self.ui = Ui_Visualization()
        self.ui.setupUi(self)

        self.setup_connections()
        self.load_data()
        self.setup_initial_state()
        self.load_saved_charts()

    def setup_connections(self):
        # Основные навигационные кнопки
        self.ui.btn_prev.clicked.connect(self.go_to_previous_page)
        self.ui.btn_next.clicked.connect(self.go_to_next_page)
        self.ui.btn_finish.clicked.connect(self.finish_visualization)

        # Простой тест если это тестовый UI
        if hasattr(self.ui, 'btn_test'):
            self.ui.btn_test.clicked.connect(self.test_function)

    def test_function(self):
        QMessageBox.information(self, "Тест", "Программа запущена успешно!")

    def setup_initial_state(self):
        self.setWindowTitle(f"DataLite - Визуализация: {self.filename}")
        if hasattr(self.ui, 'stackedWidget'):
            self.ui.stackedWidget.setCurrentIndex(0)
        self.update_navigation_buttons()
        self.populate_column_comboboxes()
        self.update_data_info()

    def load_data(self):
        try:
            file_path = f"data/storage/{self.filename}"
            print(f"Попытка загрузить файл: {file_path}")

            if not os.path.exists(file_path):
                # Создаем тестовые данные
                print("Файл не найден, создаю тестовые данные...")
                self.data = pd.DataFrame({
                    'A': [1, 2, 3, 4, 5],
                    'B': [2, 4, 6, 8, 10],
                    'C': [1, 3, 5, 7, 9],
                    'Category': ['X', 'Y', 'X', 'Y', 'X']
                })
                print(f"Созданы тестовые данные: {len(self.data)} строк")
            else:
                if self.filename.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif self.filename.endswith('.json'):
                    self.data = pd.read_json(file_path)
                else:
                    QMessageBox.critical(self, "Ошибка", "Неподдерживаемый формат файла")
                    return

            print(f"Данные загружены: {len(self.data)} строк, {len(self.data.columns)} колонок")

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            import traceback
            traceback.print_exc()
            # Создаем тестовые данные при ошибке
            self.data = pd.DataFrame({
                'X': [1, 2, 3, 4, 5],
                'Y': [2, 4, 6, 8, 10],
                'Z': [1, 3, 5, 7, 9]
            })

    def update_data_info(self):
        if self.data is not None and not self.data.empty:
            info_text = f"Данные загружены: {len(self.data)} строк, {len(self.data.columns)} колонок"
            if hasattr(self.ui, 'label_data_info'):
                self.ui.label_data_info.setText(info_text)
            print(info_text)

    def populate_column_comboboxes(self):
        if self.data is None or self.data.empty:
            return

        columns = self.data.columns.tolist()
        print(f"Колонки для заполнения: {columns}")

        # Простая реализация если UI сложный
        if hasattr(self.ui, 'line_x_combo'):
            # Заполнение всех комбобоксов
            comboboxes = [
                self.ui.line_x_combo,
                getattr(self.ui, 'pie_labels_combo', None),
                getattr(self.ui, 'pie_values_combo', None),
                getattr(self.ui, 'hist_column_combo', None),
                getattr(self.ui, 'scatter_x_combo', None),
                getattr(self.ui, 'scatter_y_combo', None),
                getattr(self.ui, 'scatter_color_combo', None),
                getattr(self.ui, 'box_category_combo', None),
                getattr(self.ui, 'box_values_combo', None),
                getattr(self.ui, 'bar_categories_combo', None),
                getattr(self.ui, 'bar_values_combo', None)
            ]

            for combo in comboboxes:
                if combo is not None:
                    combo.clear()
                    combo.addItems([""] + columns)

    def go_to_previous_page(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            if current_index > 0:
                self.ui.stackedWidget.setCurrentIndex(current_index - 1)
                self.update_navigation_buttons()

    def go_to_next_page(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            if current_index < self.ui.stackedWidget.count() - 1:
                self.ui.stackedWidget.setCurrentIndex(current_index + 1)
                self.update_navigation_buttons()

    def update_navigation_buttons(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            # Кнопка "Назад"
            if hasattr(self.ui, 'btn_prev'):
                self.ui.btn_prev.setEnabled(current_index > 0)
            # Кнопка "Далее"
            if hasattr(self.ui, 'btn_next'):
                self.ui.btn_next.setEnabled(current_index < self.ui.stackedWidget.count() - 1)

    def finish_visualization(self):
        if self.charts:
            self.chart_manager.save_charts(self.filename, self.charts)
        self.close()

    def load_saved_charts(self):
        self.charts = self.chart_manager.load_charts(self.filename)
        if self.charts:
            print(f"Загружено {len(self.charts)} сохраненных графиков")

    def closeEvent(self, event):
        if self.charts:
            self.chart_manager.save_charts(self.filename, self.charts)
        self.closed.emit()
        event.accept()
