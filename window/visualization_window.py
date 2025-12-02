import sys
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# PyQt6 imports
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QListWidgetItem,
                             QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QFrame, QStackedWidget, QCheckBox, QGridLayout,
                             QRadioButton, QMessageBox, QScrollArea,
                             QSizePolicy, QGroupBox, QAbstractItemView,
                             QFileDialog, QScrollBar)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRect
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QWheelEvent

# Matplotlib imports
import matplotlib

matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.gridspec as gridspec

# Импортируем UI из сгенерированного файла
try:
    from form.visualization import Ui_Visualization
except ImportError:
    print("Ошибка: Не удалось импортировать Ui_Visualization из form.visualization")
    print("Создаю простой UI для теста...")


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
        self.created_at = created_at or datetime.now().isoformat()
        self.id = id or f"chart_{int(datetime.now().timestamp())}_{hash(title)}"
        self.image_path = None  # Путь к временному изображению

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'chart_type': self.chart_type,
            'title': self.title,
            'data_config': self.data_config,
            'styling': self.styling,
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


class SingleChartCanvas(QWidget):
    """Виджет для отображения одного общего графика с субграфиками"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Создаем канвас для matplotlib
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # Настраиваем политику размеров для растягивания
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumWidth(800)

        # Создаем панель инструментов для навигации
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

    def update_figure(self, figure):
        """Обновляет фигуру на канвасе"""
        if self.canvas:
            # Удаляем старую фигуру
            self.canvas.deleteLater()

        # Создаем новый канвас с новой фигурой
        self.figure = figure
        self.canvas = FigureCanvas(self.figure)

        # Настраиваем политику размеров для растягивания
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)

        # Обновляем тулбар
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Обновляем layout
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Обновляем отображение
        self.canvas.draw()


class ScrollableChartWidget(QScrollArea):
    """ScrollArea с возможностью масштабирования и панорамирования"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chart_canvas = None
        self.init_ui()

    def init_ui(self):
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Создаем контейнер с растягивающимся layout
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        self.setWidget(container)

        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(5, 5, 5, 5)
        self.container_layout.setSpacing(0)

        # Создаем виджет с канвасом
        self.chart_canvas = SingleChartCanvas()
        self.chart_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.container_layout.addWidget(self.chart_canvas)

        # Устанавливаем минимальные размеры для скролла
        self.setMinimumSize(800, 400)

    def update_chart(self, figure):
        """Обновляет график в ScrollArea"""
        if self.chart_canvas:
            self.chart_canvas.update_figure(figure)

            # НАСТРОЙКИ РАЗМЕРОВ ДЛЯ КОРРЕКТНОГО ОТОБРАЖЕНИЯ:

            # 1. Рассчитываем размеры фигуры в пикселях
            fig_width_inches, fig_height_inches = figure.get_size_inches()
            fig_width_px = int(fig_width_inches * figure.dpi)
            fig_height_px = int(fig_height_inches * figure.dpi)

            # 2. Добавляем отступы для тулбара и других элементов
            toolbar_height = 40  # Примерная высота тулбара
            total_width = fig_width_px + 20  # + отступы
            total_height = fig_height_px + toolbar_height + 20

            # 3. Устанавливаем минимальные размеры контейнера
            # Это важно для корректного скроллинга
            self.widget().setMinimumSize(total_width, total_height)

            # 4. Обновляем виджет
            self.widget().updateGeometry()

            print(f"Размер фигуры: {fig_width_inches:.1f}×{fig_height_inches:.1f} дюймов")
            print(f"Размер в пикселях: {fig_width_px}×{fig_height_px}")
            print(f"Общий размер контейнера: {total_width}×{total_height}")


class UnifiedChartRenderer:
    """Рендерер, который создает одну общую фигуру со всеми субграфиками"""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def render_all_charts(self, charts: List[ChartConfig], layout_config: Dict) -> Figure:
        """Создает одну фигуру со всеми графиками в указанной компоновке"""

        rows = layout_config['rows']
        cols = layout_config['cols']
        total_charts = rows * cols

        # Адаптивные размеры фигуры в зависимости от количества графиков
        if total_charts == 1:
            # Для одного графика - больше размер
            fig_width = 10
            fig_height = 8
            dpi = 100
        elif total_charts <= 4:
            # Для 2-4 графиков
            fig_width = cols * 10
            fig_height = rows * 8
            dpi = 100
        elif total_charts <= 9:
            # Для 5-9 графиков
            fig_width = cols * 10
            fig_height = rows * 8
            dpi = 90
        else:
            # Для многих графиков
            fig_width = cols * 10
            fig_height = rows * 8
            dpi = 80

        fig = Figure(figsize=(fig_width, fig_height), dpi=dpi)

        charts_to_render = charts[:total_charts]

        for i, chart in enumerate(charts_to_render):
            if i >= total_charts:
                break

            # Создаем субграфик
            ax = fig.add_subplot(rows, cols, i + 1)
            self._render_chart_on_axes(ax, chart)

            # Добавляем заголовок субграфика
            ax.set_title(chart.title, fontsize=10 if total_charts <= 4 else 9, pad=6)

            # Включаем сетку если нужно
            if chart.styling.get('show_grid', True):
                ax.grid(True, alpha=0.3, linestyle='--')

        # Если остались пустые ячейки, скрываем их
        for i in range(len(charts_to_render), total_charts):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.axis('off')
            ax.text(0.5, 0.5, 'Пусто',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=10, color='gray')

        # Настраиваем tight_layout в зависимости от количества графиков
        if total_charts == 1:
            # Для одного графика - минимальные отступы
            fig.tight_layout(pad=2.0)
        elif total_charts <= 4:
            fig.tight_layout(pad=2.5, h_pad=3.0, w_pad=2.5)
        elif total_charts <= 9:
            fig.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
        else:
            fig.tight_layout()

        return fig

    def _render_chart_on_axes(self, ax, chart_config: ChartConfig):
        """Рендерит конкретный график на заданных осях"""
        try:
            chart_type = chart_config.chart_type
            data_config = chart_config.data_config
            styling = chart_config.styling

            if chart_type == "Линейный график (plot)":
                self._render_line_chart(ax, data_config, styling)
            elif chart_type == "Столбчатая диаграмма (bar)":
                self._render_bar_chart(ax, data_config, styling)
            elif chart_type == "Круговая диаграмма (pie)":
                self._render_pie_chart(ax, data_config, styling)
            elif chart_type == "Гистограмма (hist)":
                self._render_histogram(ax, data_config, styling)
            elif chart_type == "Диаграмма рассеяния (scatter)":
                self._render_scatter(ax, data_config, styling)
            elif chart_type == "Box Plot (boxplot)":
                self._render_boxplot(ax, data_config, styling)
            elif chart_type == "Площадной график (area)":
                self._render_area_chart(ax, data_config, styling)
            elif chart_type == "График плотности (kde)":
                self._render_kde_chart(ax, data_config, styling)

        except Exception as e:
            ax.clear()
            error_msg = f"Ошибка: {str(e)[:50]}"
            ax.text(0.5, 0.5, error_msg,
                    ha='center', va='center',
                    transform=ax.transAxes,
                    color='red', fontsize=8)
            ax.set_title(f"Ошибка", color='red', fontsize=9)
            ax.axis('off')

    def _render_line_chart(self, ax, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_cols = data_config.get('y_columns', [])

        if not x_col or not y_cols:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        if x_col not in self.data.columns:
            ax.text(0.5, 0.5, f"Колонка X '{x_col}' не найдена",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        for i, y_col in enumerate(y_cols):
            if y_col not in self.data.columns:
                continue

            line_style = styling.get('line_style', 'solid (сплошная)').split(' ')[0]
            line_width = styling.get('line_width', 2.0)
            markers = styling.get('markers', 'None (нет)').split(' ')[0]
            markersize = styling.get('markersize', 6)
            alpha = styling.get('alpha', 1.0)

            x_data = self.data[x_col]
            y_data = self.data[y_col]

            mask = x_data.notna() & y_data.notna()
            x_clean = x_data[mask]
            y_clean = y_data[mask]

            if len(x_clean) > 0:
                ax.plot(x_clean, y_clean, linestyle=line_style,
                        linewidth=line_width,
                        marker=markers if markers != 'None' else None,
                        markersize=markersize, alpha=alpha,
                        label=y_col)

        if len(y_cols) > 1 and styling.get('show_legend', True):
            ax.legend(fontsize=7, loc='upper right')

        xlabel = styling.get('xlabel', x_col)
        ylabel = styling.get('ylabel', 'Значения')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    def _render_bar_chart(self, ax, data_config: Dict, styling: Dict):
        categories_col = data_config.get('categories_column')
        values_col = data_config.get('values_column')

        if categories_col in self.data.columns and values_col in self.data.columns:
            orientation = styling.get('orientation', 'vertical (вертикальная)').split(' ')[0]
            edgecolor = styling.get('edgecolor', 'black (черный)').split(' ')[0]
            edgewidth = styling.get('edgewidth', 1.0)
            alpha = styling.get('alpha', 0.8)
            width = styling.get('width', 0.8)

            categories = self.data[categories_col].astype(str)
            values = self.data[values_col]

            if orientation == 'vertical':
                ax.bar(categories, values,
                       edgecolor=edgecolor if edgecolor != 'none' else None,
                       linewidth=edgewidth, alpha=alpha, width=width)
                ax.set_xlabel(categories_col, fontsize=9)
                ax.set_ylabel(values_col, fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                ax.barh(categories, values,
                        edgecolor=edgecolor if edgecolor != 'none' else None,
                        linewidth=edgewidth, alpha=alpha, height=width)
                ax.set_xlabel(values_col, fontsize=9)
                ax.set_ylabel(categories_col, fontsize=9)
                plt.setp(ax.get_yticklabels(), fontsize=8)

    def _render_pie_chart(self, ax, data_config: Dict, styling: Dict):
        labels_col = data_config.get('labels_column')
        values_col = data_config.get('values_column')

        if labels_col in self.data.columns and values_col in self.data.columns:
            start_angle = styling.get('start_angle', 90)
            explode = styling.get('explode', 0.1)
            autopct = styling.get('autopct', 'Не показывать')
            show_shadow = styling.get('shadow', False)
            colormap = styling.get('colormap', 'tab20c')

            labels = self.data[labels_col].astype(str)
            values = self.data[values_col]

            # Берем только первые 8 значений для читаемости
            if len(values) > 8:
                values = values[:8]
                labels = labels[:8]

            autopct_format = None
            if autopct != 'Не показывать':
                if '%1.1f%%' in autopct:
                    autopct_format = '%1.1f%%'
                elif '%1.2f%%' in autopct:
                    autopct_format = '%1.2f%%'
                elif '%d%%' in autopct:
                    autopct_format = '%d%%'

            if explode > 0:
                explode_values = [explode] + [0] * (len(values) - 1)
            else:
                explode_values = None

            try:
                cmap = plt.colormaps[colormap]
            except:
                cmap = cm.get_cmap('tab20c')

            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct=autopct_format,
                startangle=start_angle, explode=explode_values,
                shadow=show_shadow,
                textprops={'fontsize': 8}
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.axis('equal')

    def _render_histogram(self, ax, data_config: Dict, styling: Dict):
        column = data_config.get('column')

        if column in self.data.columns:
            bins = styling.get('bins', 'auto (автоматически)')
            hist_type = styling.get('hist_type', 'bar (столбчатая)').split(' ')[0]
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

            ax.hist(data, bins=bins, histtype=hist_type,
                    alpha=alpha, density=show_density, cumulative=show_cumulative,
                    edgecolor='black')

            ax.set_xlabel(column, fontsize=9)
            ax.set_ylabel('Плотность' if show_density else 'Частота', fontsize=9)

    def _render_scatter(self, ax, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_col = data_config.get('y_column')
        color_col = data_config.get('color_column')

        if x_col in self.data.columns and y_col in self.data.columns:
            point_size = styling.get('point_size', 50)
            point_alpha = styling.get('point_alpha', 0.6)
            marker = styling.get('marker', 'o (круг)').split(' ')[0]
            show_regression = styling.get('regression', False)
            colormap = styling.get('colormap', 'viridis')

            x_data = self.data[x_col]
            y_data = self.data[y_col]

            mask = x_data.notna() & y_data.notna()
            x_clean = x_data[mask]
            y_clean = y_data[mask]

            if color_col and color_col in self.data.columns:
                color_data = self.data[color_col][mask]
                scatter = ax.scatter(x_clean, y_clean, c=color_data, s=point_size / 2,
                                     alpha=point_alpha, marker=marker, cmap=colormap)
                plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                ax.scatter(x_clean, y_clean, s=point_size / 2, alpha=point_alpha,
                           marker=marker, color='blue')

            if show_regression and len(x_clean) > 1:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                ax.plot(x_clean, p(x_clean), "r--", linewidth=1, label='Линия тренда')
                ax.legend(fontsize=7)

            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)

    def _render_boxplot(self, ax, data_config: Dict, styling: Dict):
        category_col = data_config.get('category_column')
        values_col = data_config.get('values_column')

        orientation = styling.get('orientation', 'vertical (вертикальная)').split(' ')[0]
        show_points = styling.get('show_points', 'outliers (только выбросы)').split(' ')[0]
        show_notch = styling.get('notch', False)
        linewidth = styling.get('linewidth', 1.5)
        whis = styling.get('whis', 1.5)

        if category_col and category_col in self.data.columns:
            data = []
            labels = []

            for category in sorted(self.data[category_col].astype(str).unique()):
                subset = self.data[self.data[category_col].astype(str) == category][values_col]
                if len(subset) > 0:
                    data.append(subset.dropna().values)
                    labels.append(str(category))
        else:
            data = [self.data[values_col].dropna().values]
            labels = [values_col]

        showfliers = show_points in ['outliers', 'all']

        bp = ax.boxplot(data, tick_labels=labels, notch=show_notch, showfliers=showfliers,
                        patch_artist=True, boxprops=dict(linewidth=linewidth),
                        whiskerprops=dict(linewidth=linewidth), capprops=dict(linewidth=linewidth),
                        medianprops=dict(linewidth=linewidth, color='red'),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5), whis=whis)

        if orientation == 'horizontal':
            ax.invert_yaxis()

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(values_col if orientation == 'vertical' else '', fontsize=9)
        ax.set_xlabel('' if orientation == 'vertical' else values_col, fontsize=9)

    def _render_area_chart(self, ax, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_cols = data_config.get('y_columns', [])

        if x_col in self.data.columns:
            alpha = styling.get('alpha', 0.5)
            colormap = styling.get('colormap', 'viridis')

            x_data = self.data[x_col]
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
                try:
                    cmap = plt.colormaps[colormap]
                except:
                    cmap = cm.get_cmap(colormap)
                colors = cmap(np.linspace(0, 1, len(y_data_list)))

                ax.stackplot(x_data[x_data.notna()].values, *y_data_list,
                             labels=valid_cols, colors=colors, alpha=alpha)

                ax.set_xlabel(x_col, fontsize=9)
                ax.set_ylabel('Значения', fontsize=9)
                if len(valid_cols) > 1:
                    ax.legend(fontsize=7)

    def _render_kde_chart(self, ax, data_config: Dict, styling: Dict):
        columns = data_config.get('columns', [])

        alpha = styling.get('alpha', 0.5)
        colormap = styling.get('colormap', 'viridis')
        show_fill = styling.get('fill', True)
        bandwidth = styling.get('bandwidth', None)

        try:
            from scipy import stats
        except ImportError:
            ax.text(0.5, 0.5, "Для KDE установите scipy",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

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
                        ax.fill_between(x_range, y_kde, alpha=alpha * 0.7, color=colors[i])

                    ax.plot(x_range, y_kde, color=colors[i], linewidth=1.5, label=column)

        ax.set_xlabel('Значения', fontsize=9)
        ax.set_ylabel('Плотность вероятности', fontsize=9)
        if columns:
            ax.legend(fontsize=7)


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
        self.chart_renderer = None
        self.temp_dir = tempfile.mkdtemp(prefix="visualization_")

        # Ссылка на ScrollableChartWidget
        self.scrollable_chart_widget = None

        # Маппинг типов графиков на страницы
        self.chart_type_to_page = {
            "Линейный график (plot)": 1,
            "Столбчатая диаграмма (bar)": 2,
            "Круговая диаграмма (pie)": 3,
            "Гистограмма (hist)": 4,
            "Диаграмма рассеяния (scatter)": 5,
            "Box Plot (boxplot)": 6,
            "Площадной график (area)": 1,
            "График плотности (kde)": 4
        }

        self.page_to_chart_type = {
            1: "Линейный график (plot)",
            2: "Столбчатая диаграмма (bar)",
            3: "Круговая диаграмма (pie)",
            4: "Гистограмма (hist)",
            5: "Диаграмма рассеяния (scatter)",
            6: "Box Plot (boxplot)"
        }

        self.ui = Ui_Visualization()
        self.ui.setupUi(self)

        self.setup_connections()
        self.load_data()
        self.setup_initial_state()
        self.load_saved_charts()

        # Инициализация рендерера после загрузки данных
        if self.data is not None:
            self.chart_renderer = UnifiedChartRenderer(self.data)

        # Заменяем контейнер графиков на ScrollableChartWidget
        self.setup_charts_container()

    def setup_charts_container(self):
        """Настройка контейнера для отображения единого графика"""
        if hasattr(self.ui, 'gridLayout_charts'):
            # Очищаем существующий layout
            self.clear_layout(self.ui.gridLayout_charts)

            # Настраиваем свойства gridLayout для растягивания
            self.ui.gridLayout_charts.setContentsMargins(2, 2, 2, 2)
            self.ui.gridLayout_charts.setSpacing(0)

            # Создаем ScrollableChartWidget
            self.scrollable_chart_widget = ScrollableChartWidget()

            # Настраиваем политику размеров для растягивания
            self.scrollable_chart_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )

            # Добавляем его в gridLayout с растягиванием
            self.ui.gridLayout_charts.addWidget(
                self.scrollable_chart_widget,
                0, 0,  # row, column
                1, 1,  # rowSpan, columnSpan
                Qt.AlignmentFlag.AlignCenter  # Выравнивание
            )

            # Устанавливаем stretch factors для растягивания
            self.ui.gridLayout_charts.setRowStretch(0, 1)
            self.ui.gridLayout_charts.setColumnStretch(0, 1)

            print("ScrollableChartWidget создан и добавлен в контейнер")

    def clear_layout(self, layout):
        """Безопасная очистка layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self.clear_layout(sub_layout)

    def setup_connections(self):
        # Основные навигационные кнопки
        self.ui.btn_prev.clicked.connect(self.go_to_previous_page)
        self.ui.btn_next.clicked.connect(self.go_to_next_page)
        self.ui.btn_finish.clicked.connect(self.finish_visualization)

        # Кнопки на страницах графиков
        self.ui.btn_configure_chart.clicked.connect(self.configure_chart)
        self.ui.btn_line_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_line_save.clicked.connect(lambda: self.save_chart(1))
        self.ui.btn_bar_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_bar_save.clicked.connect(lambda: self.save_chart(2))
        self.ui.btn_pie_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_pie_save.clicked.connect(lambda: self.save_chart(3))
        self.ui.btn_hist_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_hist_save.clicked.connect(lambda: self.save_chart(4))
        self.ui.btn_scatter_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_scatter_save.clicked.connect(lambda: self.save_chart(5))
        self.ui.btn_box_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_box_save.clicked.connect(lambda: self.save_chart(6))

        # Кнопки на странице компоновки
        self.ui.btn_back_to_setup.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_export_figure.clicked.connect(self.export_figure)

        # Кнопки управления списком графиков
        self.ui.btn_edit_chart.clicked.connect(self.edit_chart)
        self.ui.btn_remove_chart.clicked.connect(self.remove_chart)

        # События выбора в списке графиков
        self.ui.charts_listwidget.itemSelectionChanged.connect(self.on_chart_selected)

        # Радиокнопки компоновки
        self.ui.radio_1x1.toggled.connect(lambda: self.on_layout_radio_toggled(1, 1))
        self.ui.radio_2x2.toggled.connect(lambda: self.on_layout_radio_toggled(2, 2))
        self.ui.radio_3x3.toggled.connect(lambda: self.on_layout_radio_toggled(3, 3))
        self.ui.radio_2x3.toggled.connect(lambda: self.on_layout_radio_toggled(2, 3))
        self.ui.radio_3x2.toggled.connect(lambda: self.on_layout_radio_toggled(3, 2))
        self.ui.radio_4x4.toggled.connect(lambda: self.on_layout_radio_toggled(4, 4))
        self.ui.radio_self.toggled.connect(self.on_custom_layout_toggled)

        # Спинбоксы для кастомного лейаута
        self.ui.custom_rows_spin.valueChanged.connect(self.update_custom_layout)
        self.ui.custom_cols_spin.valueChanged.connect(self.update_custom_layout)

        # Простой тест если это тестовый UI
        if hasattr(self.ui, 'btn_test'):
            self.ui.btn_test.clicked.connect(self.test_function)

    def test_function(self):
        QMessageBox.information(self, "Тест", "Программа запущена успешно!")

    def setup_initial_state(self):
        self.setWindowTitle(f"DataLite - Визуализация: {self.filename}")
        if hasattr(self.ui, 'stackedWidget'):
            self.max_page_index = self.ui.stackedWidget.count() - 1
            print(f"Всего страниц в stackedWidget: {self.max_page_index + 1}")
            self.ui.stackedWidget.setCurrentIndex(0)

        self.update_navigation_buttons()
        self.populate_column_comboboxes()
        self.update_data_info()

    def load_data(self):
        try:
            file_path = f"data/storage/{self.filename}"
            print(f"Попытка загрузить файл: {file_path}")

            if not os.path.exists(file_path):
                print("Файл не найден, создаю тестовые данные...")
                self.data = pd.DataFrame({
                    'Дата': pd.date_range(start='2023-01-01', periods=10, freq='D'),
                    'Продажи': [100, 120, 130, 110, 150, 140, 160, 170, 180, 190],
                    'Прибыль': [20, 25, 30, 22, 35, 33, 38, 40, 42, 45],
                    'Товар': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
                    'Количество': [10, 15, 12, 8, 18, 14, 16, 20, 22, 24],
                    'Цена': [10.0, 8.0, 10.8, 13.7, 8.3, 10.0, 9.9, 8.5, 8.2, 7.9],
                    'Возраст': [25, 30, 35, 28, 40, 32, 45, 38, 29, 42],
                    'Зарплата': [50000, 55000, 60000, 52000, 70000, 58000, 75000, 65000, 53000,
                                 72000]
                })
                print(f"Созданы тестовые данные: {len(self.data)} строк")
            else:
                if self.filename.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif self.filename.endswith('.json'):
                    self.data = pd.read_json(file_path)
                elif self.filename.endswith('.xlsx'):
                    self.data = pd.read_excel(file_path)
                else:
                    QMessageBox.critical(self, "Ошибка", "Неподдерживаемый формат файла")
                    return

            print(f"Данные загружены: {len(self.data)} строк, {len(self.data.columns)} колонок")

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            import traceback
            traceback.print_exc()
            self.data = pd.DataFrame({
                'Дата': pd.date_range(start='2023-01-01', periods=5, freq='D'),
                'Продажи': [100, 120, 130, 110, 150],
                'Прибыль': [20, 25, 30, 22, 35],
                'Товар': ['A', 'B', 'A', 'C', 'B'],
                'Количество': [10, 15, 12, 8, 18]
            })

    def update_data_info(self):
        if self.data is not None and not self.data.empty:
            info_text = f"Данные загружены: {len(self.data)} строк, {len(self.data.columns)} колонки"
            if hasattr(self.ui, 'label_data_info'):
                self.ui.label_data_info.setText(info_text)
            print(info_text)

    def get_suitable_columns(self, chart_type, required_numeric=True):
        """Возвращает подходящие столбцы для типа графика"""
        if self.data is None or self.data.empty:
            return []

        columns = []
        for column in self.data.columns:
            try:
                if chart_type in ["Гистограмма (hist)", "Box Plot (boxplot)",
                                  "График плотности (kde)", "Диаграмма рассеяния (scatter)"]:
                    if pd.api.types.is_numeric_dtype(self.data[column]):
                        columns.append(column)
                elif chart_type in ["Круговая диаграмма (pie)", "Столбчатая диаграмма (bar)"]:
                    columns.append(column)
                elif chart_type in ["Линейный график (plot)", "Площадной график (area)"]:
                    columns.append(column)
            except:
                columns.append(column)

        return columns

    def populate_column_comboboxes(self):
        if self.data is None or self.data.empty:
            return

        all_columns = self.data.columns.tolist()
        print(f"Все колонки: {all_columns}")

        for chart_type in self.chart_type_to_page.keys():
            suitable_columns = self.get_suitable_columns(chart_type)

            if chart_type == "Линейный график (plot)":
                if hasattr(self.ui, 'line_x_combo'):
                    self.ui.line_x_combo.clear()
                    self.ui.line_x_combo.addItems([""] + suitable_columns)
                if hasattr(self.ui, 'line_y_list'):
                    self.ui.line_y_list.clear()
                    for column in suitable_columns:
                        item = QListWidgetItem(column)
                        item.setCheckState(Qt.CheckState.Unchecked)
                        self.ui.line_y_list.addItem(item)
            elif chart_type == "Столбчатая диаграмма (bar)":
                if hasattr(self.ui, 'bar_categories_combo'):
                    self.ui.bar_categories_combo.clear()
                    self.ui.bar_categories_combo.addItems([""] + all_columns)
                if hasattr(self.ui, 'bar_values_combo'):
                    self.ui.bar_values_combo.clear()
                    numeric_cols = self.get_suitable_columns("Столбчатая диаграмма (bar)")
                    self.ui.bar_values_combo.addItems([""] + numeric_cols)
            elif chart_type == "Круговая диаграмма (pie)":
                if hasattr(self.ui, 'pie_labels_combo'):
                    self.ui.pie_labels_combo.clear()
                    self.ui.pie_labels_combo.addItems([""] + all_columns)
                if hasattr(self.ui, 'pie_values_combo'):
                    self.ui.pie_values_combo.clear()
                    numeric_cols = self.get_suitable_columns("Круговая диаграмма (pie)")
                    self.ui.pie_values_combo.addItems([""] + numeric_cols)
            elif chart_type == "Гистограмма (hist)":
                if hasattr(self.ui, 'hist_column_combo'):
                    self.ui.hist_column_combo.clear()
                    numeric_cols = self.get_suitable_columns("Гистограмма (hist)")
                    self.ui.hist_column_combo.addItems([""] + numeric_cols)
            elif chart_type == "Диаграмма рассеяния (scatter)":
                if hasattr(self.ui, 'scatter_x_combo'):
                    self.ui.scatter_x_combo.clear()
                    numeric_cols = self.get_suitable_columns("Диаграмма рассеяния (scatter)")
                    self.ui.scatter_x_combo.addItems([""] + numeric_cols)
                if hasattr(self.ui, 'scatter_y_combo'):
                    self.ui.scatter_y_combo.clear()
                    self.ui.scatter_y_combo.addItems([""] + numeric_cols)
                if hasattr(self.ui, 'scatter_color_combo'):
                    self.ui.scatter_color_combo.clear()
                    self.ui.scatter_color_combo.addItems(["Нет"] + all_columns)
            elif chart_type == "Box Plot (boxplot)":
                if hasattr(self.ui, 'box_category_combo'):
                    self.ui.box_category_combo.clear()
                    self.ui.box_category_combo.addItems(["Нет (один бокс)"] + all_columns)
                if hasattr(self.ui, 'box_values_combo'):
                    self.ui.box_values_combo.clear()
                    numeric_cols = self.get_suitable_columns("Box Plot (boxplot)")
                    self.ui.box_values_combo.addItems([""] + numeric_cols)

    def go_to_previous_page(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            if 0 < current_index <= 7:
                self.ui.stackedWidget.setCurrentIndex(0)
            elif current_index > 0:
                self.ui.stackedWidget.setCurrentIndex(current_index - 1)
            self.update_navigation_buttons()

    def go_to_next_page(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            max_index = self.ui.stackedWidget.count() - 1

            if current_index == 0:
                if not self.charts:
                    QMessageBox.warning(self, "Внимание", "Сначала создайте хотя бы один график!")
                    return
                self.go_to_page(7)
            elif 7 <= current_index < max_index:
                self.ui.stackedWidget.setCurrentIndex(current_index + 1)
                self.update_navigation_buttons()

    def go_to_page(self, page_index):
        """Переход на конкретную страницу"""
        if hasattr(self.ui, 'stackedWidget'):
            if 0 <= page_index < self.ui.stackedWidget.count():
                self.ui.stackedWidget.setCurrentIndex(page_index)
                self.update_navigation_buttons()
                print(f"Перешли на страницу {page_index}")
            else:
                print(f"Ошибка: страницы {page_index} не существует")

    def update_navigation_buttons(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            print(f"Текущая страница: {current_index}")

            if hasattr(self.ui, 'btn_prev'):
                self.ui.btn_prev.setEnabled(current_index > 0)

            if hasattr(self.ui, 'btn_next'):
                if current_index == 0:
                    self.ui.btn_next.setText("К компоновке")
                    self.ui.btn_next.clicked.disconnect()
                    self.ui.btn_next.clicked.connect(self.go_to_next_page)
                    self.ui.btn_next.setEnabled(True)
                elif current_index == 7:
                    self.ui.btn_next.setText("Показать графики")
                    self.ui.btn_next.clicked.disconnect()
                    self.ui.btn_next.clicked.connect(self.generate_layout)
                    self.ui.btn_next.setEnabled(len(self.charts) > 0)
                elif current_index == 8:
                    self.ui.btn_next.setEnabled(False)
                    self.ui.btn_next.setText("Далее")
                else:
                    self.ui.btn_next.setEnabled(False)
                    self.ui.btn_next.setText("Далее(недоступно)")

            if hasattr(self.ui, 'btn_finish'):
                self.ui.btn_finish.setEnabled(current_index in [0, 8])

    def configure_chart(self):
        """Настройка параметров графика - переход на страницу конкретного типа графика"""
        chart_type = self.ui.chart_type_combo.currentText()

        if not chart_type:
            QMessageBox.warning(self, "Внимание", "Выберите тип графика!")
            return

        title = self.ui.chart_title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, "Внимание", "Введите название графика!")
            return

        page_index = self.chart_type_to_page.get(chart_type, 1)

        if page_index >= self.ui.stackedWidget.count():
            QMessageBox.warning(self, "Ошибка",
                                f"Страница настройки для типа '{chart_type}' не найдена")
            return

        self.go_to_page(page_index)

    def save_chart(self, page_type):
        """Сохранение конфигурации графика"""
        try:
            chart_type = ""
            data_config = {}
            styling = {}

            title = self.ui.chart_title_edit.text().strip()
            if not title:
                QMessageBox.warning(self, "Внимание", "Введите название графика!")
                return

            if page_type == 1:
                chart_type = "Линейный график (plot)"

                x_col = self.ui.line_x_combo.currentText()
                if not x_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбец для оси X!")
                    return

                y_cols = []
                for i in range(self.ui.line_y_list.count()):
                    item = self.ui.line_y_list.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        y_cols.append(item.text())

                if not y_cols:
                    QMessageBox.warning(self, "Внимание",
                                        "Выберите хотя бы один столбец для оси Y!")
                    return

                data_config = {
                    'x_column': x_col,
                    'y_columns': y_cols
                }

                styling = {
                    'line_style': self.ui.line_style_combo.currentText(),
                    'line_width': self.ui.line_width_spin.value(),
                    'markers': self.ui.markers_combo.currentText(),
                    'markersize': self.ui.markersize_spin.value(),
                    'alpha': self.ui.line_alpha_spin.value(),
                    'show_grid': self.ui.grid_checkbox.isChecked(),
                    'show_legend': True,
                    'xlabel': self.ui.xlabel_edit.text() or x_col,
                    'ylabel': self.ui.ylabel_edit.text() or 'Значения'
                }

            elif page_type == 2:
                chart_type = "Столбчатая диаграмма (bar)"

                categories_col = self.ui.bar_categories_combo.currentText()
                values_col = self.ui.bar_values_combo.currentText()

                if not categories_col or not values_col:
                    QMessageBox.warning(self, "Внимание",
                                        "Выберите столбцы для категорий и значений!")
                    return

                data_config = {
                    'categories_column': categories_col,
                    'values_column': values_col
                }

                styling = {
                    'orientation': self.ui.bar_orientation_combo.currentText(),
                    'edgecolor': self.ui.bar_edgecolor_combo.currentText(),
                    'edgewidth': self.ui.bar_edgewidth_spin.value(),
                    'alpha': self.ui.bar_alpha_spin.value(),
                    'width': self.ui.bar_width_spin.value()
                }

            elif page_type == 3:
                chart_type = "Круговая диаграмма (pie)"

                labels_col = self.ui.pie_labels_combo.currentText()
                values_col = self.ui.pie_values_combo.currentText()

                if not labels_col or not values_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбцы для меток и значений!")
                    return

                data_config = {
                    'labels_column': labels_col,
                    'values_column': values_col
                }

                styling = {
                    'start_angle': self.ui.pie_start_angle_spin.value(),
                    'explode': self.ui.pie_explode_spin.value(),
                    'autopct': self.ui.pie_autopct_combo.currentText(),
                    'shadow': self.ui.pie_shadow_checkbox.isChecked(),
                }

            elif page_type == 4:
                chart_type = "Гистограмма (hist)"

                column = self.ui.hist_column_combo.currentText()
                if not column:
                    QMessageBox.warning(self, "Внимание", "Выберите столбец для гистограммы!")
                    return

                data_config = {
                    'column': column
                }

                styling = {
                    'bins': self.ui.bins_combo.currentText(),
                    'hist_type': self.ui.hist_type_combo.currentText(),
                    'alpha': self.ui.hist_alpha_spin.value(),
                    'density': self.ui.hist_density_checkbox.isChecked(),
                    'cumulative': self.ui.hist_cumulative_checkbox.isChecked()
                }

            elif page_type == 5:
                chart_type = "Диаграмма рассеяния (scatter)"

                x_col = self.ui.scatter_x_combo.currentText()
                y_col = self.ui.scatter_y_combo.currentText()

                if not x_col or not y_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбцы для осей X и Y!")
                    return

                data_config = {
                    'x_column': x_col,
                    'y_column': y_col
                }

                color_col = self.ui.scatter_color_combo.currentText()
                if color_col and color_col != "Нет":
                    data_config['color_column'] = color_col

                styling = {
                    'point_size': self.ui.point_size_spin.value(),
                    'point_alpha': self.ui.point_alpha_spin.value(),
                    'marker': self.ui.scatter_marker_combo.currentText(),
                    'regression': self.ui.regression_checkbox.isChecked(),
                    'colormap': self.ui.scatter_colormap_combo.currentText()
                }

            elif page_type == 6:
                chart_type = "Box Plot (boxplot)"

                values_col = self.ui.box_values_combo.currentText()
                if not values_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбец значений!")
                    return

                data_config = {
                    'values_column': values_col
                }

                category_col = self.ui.box_category_combo.currentText()
                if category_col and category_col != "Нет (один бокс)":
                    data_config['category_column'] = category_col

                styling = {
                    'orientation': self.ui.box_orientation_combo.currentText(),
                    'show_points': self.ui.box_show_points_combo.currentText(),
                    'notch': self.ui.box_notch_checkbox.isChecked(),
                    'color': self.ui.box_color_combo.currentText(),
                    'linewidth': self.ui.box_linewidth_spin.value(),
                    'whis': self.ui.box_whis_spin.value()
                }

            chart_config = ChartConfig(
                chart_type=chart_type,
                title=title,
                data_config=data_config,
                styling=styling
            )

            self.charts.append(chart_config)
            self.update_charts_list()
            self.chart_manager.save_charts(self.filename, self.charts)
            self.go_to_page(0)

            QMessageBox.information(self, "Успех", f"График '{title}' сохранен!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении графика: {str(e)}")
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()

    def update_charts_list(self):
        """Обновление списка графиков в QListWidget"""
        if hasattr(self.ui, 'charts_listwidget'):
            self.ui.charts_listwidget.clear()

            for i, chart in enumerate(self.charts):
                item_text = f"{i + 1}. {chart.title} ({chart.chart_type})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.ui.charts_listwidget.addItem(item)

    def on_chart_selected(self):
        """Обработка выбора графика в списке"""
        selected_items = self.ui.charts_listwidget.selectedItems()
        if selected_items:
            self.ui.btn_edit_chart.setEnabled(True)
            self.ui.btn_remove_chart.setEnabled(True)
        else:
            self.ui.btn_edit_chart.setEnabled(False)
            self.ui.btn_remove_chart.setEnabled(False)

    def edit_chart(self):
        """Редактирование выбранного графика"""
        selected_items = self.ui.charts_listwidget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        chart_index = item.data(Qt.ItemDataRole.UserRole)

        if 0 <= chart_index < len(self.charts):
            chart = self.charts[chart_index]

            self.ui.chart_type_combo.setCurrentText(chart.chart_type)
            self.ui.chart_title_edit.setText(chart.title)

            chart_type = chart.chart_type
            page_index = self.chart_type_to_page.get(chart_type, 1)
            self.go_to_page(page_index)

            if chart_type == "Линейный график (plot)":
                self._fill_line_chart_fields(chart)
            elif chart_type == "Столбчатая диаграмма (bar)":
                self._fill_bar_chart_fields(chart)
            elif chart_type == "Круговая диаграмма (pie)":
                self._fill_pie_chart_fields(chart)
            elif chart_type == "Гистограмма (hist)":
                self._fill_histogram_fields(chart)
            elif chart_type == "Диаграмма рассеяния (scatter)":
                self._fill_scatter_fields(chart)
            elif chart_type == "Box Plot (boxplot)":
                self._fill_boxplot_fields(chart)

            del self.charts[chart_index]
            self.update_charts_list()

            QMessageBox.information(self, "Редактирование",
                                    f"График '{chart.title}' готов к редактированию. Настройте параметры и сохраните.")

    def _fill_line_chart_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        x_col = data_config.get('x_column', '')
        y_cols = data_config.get('y_columns', [])

        self.ui.line_x_combo.setCurrentText(x_col)

        for i in range(self.ui.line_y_list.count()):
            item = self.ui.line_y_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in y_cols else Qt.CheckState.Unchecked)

        self.ui.line_style_combo.setCurrentText(styling.get('line_style', 'solid (сплошная)'))
        self.ui.line_width_spin.setValue(styling.get('line_width', 2.0))
        self.ui.markers_combo.setCurrentText(styling.get('markers', 'None (нет)'))
        self.ui.markersize_spin.setValue(styling.get('markersize', 6))
        self.ui.line_alpha_spin.setValue(styling.get('alpha', 1.0))
        self.ui.grid_checkbox.setChecked(styling.get('show_grid', True))
        self.ui.xlabel_edit.setText(styling.get('xlabel', ''))
        self.ui.ylabel_edit.setText(styling.get('ylabel', ''))

    def _fill_bar_chart_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.bar_categories_combo.setCurrentText(data_config.get('categories_column', ''))
        self.ui.bar_values_combo.setCurrentText(data_config.get('values_column', ''))

        self.ui.bar_orientation_combo.setCurrentText(
            styling.get('orientation', 'vertical (вертикальная)'))
        self.ui.bar_edgecolor_combo.setCurrentText(styling.get('edgecolor', 'black (черный)'))
        self.ui.bar_edgewidth_spin.setValue(styling.get('edgewidth', 1.0))
        self.ui.bar_alpha_spin.setValue(styling.get('alpha', 0.8))
        self.ui.bar_width_spin.setValue(styling.get('width', 0.8))

    def _fill_pie_chart_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.pie_labels_combo.setCurrentText(data_config.get('labels_column', ''))
        self.ui.pie_values_combo.setCurrentText(data_config.get('values_column', ''))

        self.ui.pie_start_angle_spin.setValue(styling.get('start_angle', 90))
        self.ui.pie_explode_spin.setValue(styling.get('explode', 0.1))
        self.ui.pie_autopct_combo.setCurrentText(styling.get('autopct', 'Не показывать'))
        self.ui.pie_shadow_checkbox.setChecked(styling.get('shadow', False))

    def _fill_histogram_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.hist_column_combo.setCurrentText(data_config.get('column', ''))

        self.ui.bins_combo.setCurrentText(styling.get('bins', 'auto (автоматически)'))
        self.ui.hist_type_combo.setCurrentText(styling.get('hist_type', 'bar (столбчатая)'))
        self.ui.hist_alpha_spin.setValue(styling.get('alpha', 0.7))
        self.ui.hist_density_checkbox.setChecked(styling.get('density', False))
        self.ui.hist_cumulative_checkbox.setChecked(styling.get('cumulative', False))

    def _fill_scatter_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.scatter_x_combo.setCurrentText(data_config.get('x_column', ''))
        self.ui.scatter_y_combo.setCurrentText(data_config.get('y_column', ''))

        color_col = data_config.get('color_column', '')
        self.ui.scatter_color_combo.setCurrentText(color_col if color_col else "Нет")

        self.ui.point_size_spin.setValue(styling.get('point_size', 50))
        self.ui.point_alpha_spin.setValue(styling.get('point_alpha', 0.6))
        self.ui.scatter_marker_combo.setCurrentText(styling.get('marker', 'o (круг)'))
        self.ui.regression_checkbox.setChecked(styling.get('regression', False))
        self.ui.scatter_colormap_combo.setCurrentText(styling.get('colormap', 'viridis'))

    def _fill_boxplot_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.box_values_combo.setCurrentText(data_config.get('values_column', ''))

        category_col = data_config.get('category_column', '')
        self.ui.box_category_combo.setCurrentText(
            category_col if category_col else "Нет (один бокс)")

        self.ui.box_orientation_combo.setCurrentText(
            styling.get('orientation', 'vertical (вертикальная)'))
        self.ui.box_show_points_combo.setCurrentText(
            styling.get('show_points', 'outliers (только выбросы)'))
        self.ui.box_notch_checkbox.setChecked(styling.get('notch', False))
        self.ui.box_color_combo.setCurrentText(styling.get('color', 'lightblue'))
        self.ui.box_linewidth_spin.setValue(styling.get('linewidth', 1.5))
        self.ui.box_whis_spin.setValue(styling.get('whis', 1.5))

    def remove_chart(self):
        """Удаление выбранного графика"""
        selected_items = self.ui.charts_listwidget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        chart_index = item.data(Qt.ItemDataRole.UserRole)

        if 0 <= chart_index < len(self.charts):
            chart = self.charts[chart_index]

            reply = QMessageBox.question(self, "Подтверждение",
                                         f"Вы уверены, что хотите удалить график '{chart.title}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                del self.charts[chart_index]
                self.update_charts_list()
                self.chart_manager.save_charts(self.filename, self.charts)

                self.ui.btn_edit_chart.setEnabled(False)
                self.ui.btn_remove_chart.setEnabled(False)

    def on_layout_radio_toggled(self, rows, cols):
        """Обработка выбора стандартной компоновки"""
        if self.sender().isChecked():
            self.layout_config = {'rows': rows, 'cols': cols}
            self.ui.custom_rows_spin.setEnabled(False)
            self.ui.custom_cols_spin.setEnabled(False)


    def on_custom_layout_toggled(self, checked):
        """Обработка выбора кастомной компоновки"""
        if checked:
            self.ui.custom_rows_spin.setEnabled(True)
            self.ui.custom_cols_spin.setEnabled(True)
            self.update_custom_layout()

    def update_custom_layout(self):
        """Обновление кастомного лейаута"""
        if self.ui.radio_self.isChecked():
            rows = self.ui.custom_rows_spin.value()
            cols = self.ui.custom_cols_spin.value()
            self.layout_config = {'rows': rows, 'cols': cols}

            if hasattr(self.ui, 'current_layout_label'):
                total_charts = rows * cols
                self.ui.current_layout_label.setText(
                    f"Компоновка: {rows} × {cols} | Графиков: {total_charts}")

    def generate_layout(self):
        """Генерация единого графика со всеми субграфиками"""
        print("Нажата кнопка 'Построить все графики'")

        if not self.charts:
            QMessageBox.warning(self, "Внимание", "Нет сохраненных графиков для отображения!")
            return

        if not self.chart_renderer:
            self.chart_renderer = UnifiedChartRenderer(self.data)

        try:
            # Создаем единую фигуру со всеми графиками
            figure = self.chart_renderer.render_all_charts(self.charts, self.layout_config)

            # Отображаем фигуру в ScrollableChartWidget
            if self.scrollable_chart_widget:
                self.scrollable_chart_widget.update_chart(figure)

                # Обновляем информацию о компоновке
                if hasattr(self.ui, 'current_layout_label'):
                    rows = self.layout_config['rows']
                    cols = self.layout_config['cols']
                    displayed = min(len(self.charts), rows * cols)
                    self.ui.current_layout_label.setText(
                        f"Компоновка: {rows} × {cols} | Отображено графиков: {displayed}/{len(self.charts)}")

                # Переходим на страницу отображения
                self.go_to_page(8)

                QMessageBox.information(self, "Успех",
                                        f"Графики построены! Отображено {displayed} из {len(self.charts)} графиков.")
            else:
                QMessageBox.critical(self, "Ошибка",
                                     "Виджет для отображения графиков не инициализирован!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при построении графиков: {str(e)}")
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()

    def export_figure(self):
        """Экспорт графика в файл"""
        if not self.scrollable_chart_widget or not self.scrollable_chart_widget.chart_canvas:
            QMessageBox.warning(self, "Внимание", "Нет графика для экспорта!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )

        if file_path and self.scrollable_chart_widget.chart_canvas.figure:
            try:
                # Сохраняем фигуру
                self.scrollable_chart_widget.chart_canvas.figure.savefig(
                    file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Успех", f"График сохранен в {file_path}!")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении графика: {str(e)}")
                print(f"Ошибка экспорта: {e}")

    def finish_visualization(self):
        """Завершение работы с визуализацией"""
        if self.charts:
            self.chart_manager.save_charts(self.filename, self.charts)
        self.close()

    def load_saved_charts(self):
        """Загрузка сохраненных графиков"""
        self.charts = self.chart_manager.load_charts(self.filename)
        if self.charts:
            print(f"Загружено {len(self.charts)} сохраненных графиков")
            self.update_charts_list()

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.charts:
            self.chart_manager.save_charts(self.filename, self.charts)

        # Очистка временных файлов
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Очищена временная директория: {self.temp_dir}")
            except Exception as e:
                print(f"Ошибка при очистке временной директории: {e}")

        self.closed.emit()
        event.accept()

    def resizeEvent(self, event):
        """Обработка изменения размера окна"""
        super().resizeEvent(event)

        # При изменении размера окна обновляем отображение графика
        if hasattr(self, 'scrollable_chart_widget') and self.scrollable_chart_widget:
            if self.scrollable_chart_widget.chart_canvas and \
                    self.scrollable_chart_widget.chart_canvas.canvas:
                self.scrollable_chart_widget.chart_canvas.canvas.draw()


# Для тестирования модуля
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Тестовый файл
    test_filename = "test_data.csv"

    # Создаем тестовые данные если их нет
    os.makedirs("data/storage", exist_ok=True)
    test_data = pd.DataFrame({
        'Дата': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Продажи': [100, 120, 130, 110, 150, 140, 160, 170, 180, 190],
        'Прибыль': [20, 25, 30, 22, 35, 33, 38, 40, 42, 45],
        'Товар': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'Количество': [10, 15, 12, 8, 18, 14, 16, 20, 22, 24],
        'Цена': [10.0, 8.0, 10.8, 13.7, 8.3, 10.0, 9.9, 8.5, 8.2, 7.9]
    })
    test_data.to_csv(f"data/storage/{test_filename}", index=False)

    window = VisualizationWindow(test_filename)
    window.show()

    sys.exit(app.exec())