import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, chi2_contingency
import seaborn as sns

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QListWidget, QPushButton,
    QSpinBox, QTableView, QTextEdit, QFrame,
    QScrollArea, QHeaderView, QMessageBox,
    QAbstractItemView, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_percentage_error,
    mean_absolute_error, mean_squared_error
)
import matplotlib
matplotlib.use('QtAgg')
# Импортируем скомпилированный UI
from form.modeling_window_ui import Ui_ModelingWindow


class ModelingWindow(QMainWindow):
    """Окно для построения моделей машинного обучения"""

    # Сигнал для закрытия окна
    closed = pyqtSignal()

    def __init__(self, filename, parent=None):
        super().__init__(parent)

        # Инициализация UI
        self.ui = Ui_ModelingWindow()
        self.ui.setupUi(self)

        # Сохраняем параметры
        self.filename = filename
        self.parent = parent
        self.data_folder = parent.data_folder if parent else "data"

        # Инициализация переменных
        self.df = None
        self.model = None
        self.ohe = OneHotEncoder(sparse_output=False, drop='first')
        self.categorical_features = []
        self.numerical_features = []

        # Настройка интерфейса
        self.setup_ui()

        # Загрузка данных
        self.load_data()

        # Подключение сигналов
        self.connect_signals()

    def setup_ui(self):
        """Настройка элементов интерфейса"""
        # Установка списка моделей
        self.ui.modelComboBox.addItems(["LinearRegression"])

        # Настройка таблицы предсказаний
        self.predictions_model = QStandardItemModel()
        self.predictions_model.setHorizontalHeaderLabels(['Фактическое', 'Предсказанное'])
        self.ui.predictionsTableView.setModel(self.predictions_model)

        # Настройка размеров столбцов
        header = self.ui.predictionsTableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Центрирование окна
        self.center_window()

        # Установка размера окна
        self.resize(1200, 800)

    def center_window(self):
        """Центрирование окна на экране"""
        screen = self.screen().availableGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    def load_data(self):
        """Загрузка данных из файла"""
        try:
            file_path = os.path.join(self.data_folder, self.filename)

            # Проверяем, есть ли информация о разделителе в file_states.csv
            separator = self.get_file_separator()

            # Загружаем данные с правильным разделителем
            if separator:
                self.df = pd.read_csv(file_path, sep=separator)
                print(f"Загружены данные с разделителем: '{separator}'")
            else:
                # Пробуем разные разделители
                self.df = self.try_different_separators(file_path)

            # Определяем типы колонок
            self.identify_column_types()

            # Заполняем выпадающие списки
            self.populate_selection_lists()

            # Устанавливаем целевой по умолчанию
            self.set_default_target()

            print(f"Успешно загружено: {len(self.df)} строк, {len(self.df.columns)} столбцов")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {str(e)}")
            self.close()

    def get_file_separator(self):
        """Получение разделителя из файла file_states.csv"""
        try:
            states_file = os.path.join(self.data_folder, "app_data", "file_states.csv")

            if os.path.exists(states_file):
                states_df = pd.read_csv(states_file)

                # Ищем запись для текущего файла
                file_record = states_df[states_df['name'] == self.filename]

                if not file_record.empty:
                    separator = file_record.iloc[0]['separator']
                    # Преобразуем строковые escape-последовательности в реальные символы
                    if separator == '\\t' or separator == 'tab':
                        return '\t'
                    elif separator == ',':
                        return ','
                    elif separator == ';':
                        return ';'
                    elif separator == '|':
                        return '|'
                    elif separator == '\\s+':
                        return r'\s+'
                    else:
                        # Пробуем использовать как есть
                        return separator
            return None

        except Exception as e:
            print(f"Ошибка при чтении file_states.csv: {e}")
            return None

    def try_different_separators(self, file_path):
        """Попробовать загрузить файл с разными разделителями"""
        separators = [';', ',', '\t', '|', ' ', r'\s+']

        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep, engine='python')
                # Проверяем, что загрузилось больше одной колонки
                if len(df.columns) > 1:
                    print(f"Найден разделитель: '{sep}'")
                    return df
            except Exception as e:
                continue

        # Если ничего не помогло, пробуем загрузить с разделителем по умолчанию
        try:
            print("Использую разделитель по умолчанию ','")
            return pd.read_csv(file_path)
        except Exception as e:
            # Последняя попытка - загрузить с параметром error_bad_lines=False
            try:
                return pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
            except:
                raise Exception(f"Не удалось загрузить файл {self.filename} с любым разделителем")

    def identify_column_types(self):
        """Определение типов колонок (категориальные/числовые)"""
        self.categorical_features = []
        self.numerical_features = []

        if self.df is None:
            return

        for column in self.df.columns:
            # Проверяем на строки, которые могут быть числами
            if self.df[column].dtype in ['object', 'bool', 'category']:
                # Пробуем преобразовать в числа, если это возможно
                try:
                    # Пробуем преобразовать в числовой тип
                    numeric_col = pd.to_numeric(self.df[column], errors='coerce')
                    # Если более 70% значений удалось преобразовать, считаем числовым
                    if numeric_col.notna().sum() / len(numeric_col) > 0.7:
                        self.numerical_features.append(column)
                        self.df[column] = numeric_col
                    else:
                        self.categorical_features.append(column)
                except:
                    self.categorical_features.append(column)
            elif pd.api.types.is_numeric_dtype(self.df[column]):
                self.numerical_features.append(column)
            else:
                # Для всех остальных типов считаем категориальными
                self.categorical_features.append(column)

        print(f"Найдено числовых признаков: {len(self.numerical_features)}")
        print(f"Найдено категориальных признаков: {len(self.categorical_features)}")
        print(f"Числовые: {self.numerical_features}")
        print(f"Категориальные: {self.categorical_features}")

    def populate_selection_lists(self):
        """Заполнение списков выбора"""
        # Целевая переменная
        self.ui.targetComboBox.clear()
        self.ui.targetComboBox.addItems(self.df.columns.tolist())

        # Список признаков
        self.ui.featuresListWidget.clear()
        self.ui.featuresListWidget.addItems(self.df.columns.tolist())

        # Автоматически выбираем все признаки кроме первого
        for i in range(1, self.ui.featuresListWidget.count()):
            item = self.ui.featuresListWidget.item(i)
            item.setSelected(True)

    def set_default_target(self):
        """Установка целевой переменной по умолчанию"""
        # Пытаемся найти подходящую целевую переменную
        possible_targets = []

        for col in self.numerical_features:
            # Проверяем, что в колонке нет пропусков и это не индекс
            if not self.df[col].isnull().any() and len(self.df[col].unique()) > 10:
                possible_targets.append(col)

        if possible_targets:
            # Выбираем первую подходящую целевую
            default_target = possible_targets[0]
            index = self.ui.targetComboBox.findText(default_target)
            if index >= 0:
                self.ui.targetComboBox.setCurrentIndex(index)

    def connect_signals(self):
        """Подключение сигналов к слотам"""
        self.ui.showHeatmapButton.clicked.connect(self.show_heatmap)
        self.ui.buildModelButton.clicked.connect(self.build_model)
        self.ui.closeButton.clicked.connect(self.close)

    def correlation_ratio(self, categories, values):
        """Вычисляет корреляционное отношение между категориальной и количественной переменными"""
        categories = np.array(categories)
        values = np.array(values)
        ssw = 0
        ssb = 0
        for category in set(categories):
            subgroup = values[np.where(categories == category)[0]]
            ssw += sum((subgroup - np.mean(subgroup)) ** 2)
            ssb += len(subgroup) * (np.mean(subgroup) - np.mean(values)) ** 2
        if ssb + ssw == 0:
            return np.nan
        return round((ssb / (ssb + ssw)) ** 0.5, 4)

    def my_cramers(self, x, y):
        """Вычисляет коэффициент корреляции Крамера между двумя категориальными переменными"""
        data = pd.crosstab(x, y)
        n = data.sum().sum()
        theory = np.outer(data.values.sum(axis=1),
                          data.values.sum(axis=0)) / n
        chi2 = ((data.values - theory) ** 2 / theory).sum()
        chi2 = chi2_contingency(data)[0]
        r, c = data.values.shape
        if min(r - 1, c - 1) == 0:
            cramer = 0
        else:
            cramer = np.sqrt(chi2 / (n * min(r - 1, c - 1)))
        return round(cramer, 4)

    def show_heatmap(self):
        """Отображение тепловой карты корреляций"""
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены!")
            return

        try:
            # Создаем матрицу корреляций
            corr_df = pd.DataFrame(index=self.df.columns, columns=self.df.columns)

            for col1 in self.df.columns:
                for col2 in self.df.columns:
                    if col1 == col2:
                        corr_df.loc[col1, col2] = 1.0
                    else:
                        if (self.df[col1].dtype in ('int64', 'float64') and
                                self.df[col2].dtype in ('int64', 'float64')):
                            dt1 = self.df[col1]
                            dt2 = self.df[col2]
                            try:
                                if (shapiro(dt1)[1] >= 0.05 and
                                        shapiro(dt2)[1] >= 0.05):
                                    corr_df.loc[col1, col2] = self.df[[col1, col2]].corr().iloc[
                                        0, 1]
                                else:
                                    corr_df.loc[col1, col2] = \
                                        self.df[[col1, col2]].corr(method='spearman').iloc[0, 1]
                            except:
                                corr_df.loc[col1, col2] = \
                                    self.df[[col1, col2]].corr(method='spearman').iloc[0, 1]
                        elif (self.df[col1].dtype in ['object', 'bool'] and
                              self.df[col2].dtype in ['object', 'bool']):
                            dt1 = self.df[col1]
                            dt2 = self.df[col2]
                            cramer = self.my_cramers(dt1, dt2)
                            corr_df.loc[col1, col2] = round(cramer, 2)
                        else:
                            g = self.df.dropna(subset=[col1, col2])
                            if self.df[col1].dtype in ['object', 'bool']:
                                corr_df.loc[col1, col2] = self.correlation_ratio(g[col1], g[col2])
                            elif self.df[col2].dtype in ['object', 'bool']:
                                corr_df.loc[col1, col2] = self.correlation_ratio(g[col2], g[col1])

            # Преобразуем в float
            for col in corr_df.columns:
                corr_df[col] = corr_df[col].astype('float64')

            # Отображаем тепловую карту
            n_cols = len(corr_df.columns)

            # Автоматически выбираем параметры
            if n_cols > 25:
                # Для большого количества колонок не показываем аннотации
                show_annot = False
                figsize = (20, 18)
                label_size = 8
            elif n_cols > 15:
                show_annot = True
                figsize = (16, 14)
                label_size = 9
                annot_size = 8
            else:
                show_annot = True
                figsize = (14, 12)
                label_size = 11
                annot_size = 10

            plt.figure(figsize=figsize)

            mask = np.triu(np.ones_like(corr_df, dtype=bool))

            # Строим heatmap с аннотациями или без
            if show_annot:
                ax = sns.heatmap(corr_df, mask=mask, cmap='coolwarm', annot=True,
                                 fmt='.2f', annot_kws={'size': annot_size},
                                 square=True, cbar_kws={"shrink": 0.8}, center=0,
                                 linewidths=0.5, linecolor='white')
            else:
                ax = sns.heatmap(corr_df, mask=mask, cmap='coolwarm', annot=False,
                                 square=True, cbar_kws={"shrink": 0.8}, center=0,
                                 linewidths=0.5, linecolor='white')

            plt.title('Матрица корреляций', fontsize=16, fontweight='bold', pad=20)

            # Настраиваем метки осей
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=label_size)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_size)

            plt.tight_layout()
            plt.show(block=True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить тепловую карту: {str(e)}")

    def get_ohe(self, train, categ):
        """Преобразует категориальные признаки в one-hot encoding"""
        temp_df = pd.DataFrame(
            data=self.ohe.transform(train[categ]),
            columns=self.ohe.get_feature_names_out()
        )
        data = pd.concat([train.reset_index(drop=True), temp_df], axis=1)
        data = data.drop(columns=categ, axis=1)
        return data

    def calculate_metrics(self, fact, prediction):
        """Вычисление метрик модели"""
        metrics = {}
        try:
            metrics['R2'] = round(r2_score(fact, prediction), 4)
        except:
            metrics['R2'] = 0.0

        try:
            metrics['MAPE'] = round(mean_absolute_percentage_error(fact, prediction) * 100, 3)
        except:
            metrics['MAPE'] = 0.0

        metrics['MAE'] = round(mean_absolute_error(fact, prediction), 4)
        metrics['RMSE'] = round(mean_squared_error(fact, prediction) ** 0.5, 4)
        return metrics

    def build_model(self):
        """Построение модели"""
        try:
            # Получаем выбранные параметры
            target = self.ui.targetComboBox.currentText()
            test_size = self.ui.testSizeSpinBox.value() / 100
            random_state = self.ui.randomSeedSpinBox.value()

            # Получаем выбранные признаки
            selected_features = [
                self.ui.featuresListWidget.item(i).text()
                for i in range(self.ui.featuresListWidget.count())
                if self.ui.featuresListWidget.item(i).isSelected()
            ]

            # Проверки
            if not selected_features:
                QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один признак!")
                return

            if target in selected_features:
                QMessageBox.warning(self, "Ошибка",
                                    "Целевая переменная не может быть среди признаков!")
                return

            # Проверяем наличие пропусков в целевой переменной
            if self.df[target].isnull().any():
                QMessageBox.warning(self, "Ошибка",
                                    f"В целевой переменной '{target}' есть пропущенные значения!")
                return

            # Определяем категориальные и числовые признаки
            categorical_features = [f for f in selected_features if f in self.categorical_features]
            numerical_features = [f for f in selected_features if f in self.numerical_features]

            # Удаляем пропуски
            df_clean = self.df[selected_features + [target]].dropna()

            if len(df_clean) == 0:
                QMessageBox.warning(self, "Ошибка",
                                    "После удаления пропусков данные отсутствуют!")
                return

            if len(df_clean) < 10:
                QMessageBox.warning(self, "Ошибка",
                                    f"Слишком мало данных после очистки: {len(df_clean)} строк")
                return

            # Разделяем данные
            X_train, X_test, y_train, y_test = train_test_split(
                df_clean[selected_features],
                df_clean[target],
                test_size=test_size,
                random_state=random_state
            )

            # Обрабатываем категориальные признаки
            if categorical_features:
                try:
                    self.ohe.fit(X_train[categorical_features])
                    X_train = self.get_ohe(X_train, categorical_features)
                    X_test = self.get_ohe(X_test, categorical_features)
                except Exception as e:
                    QMessageBox.warning(self, "Предупреждение",
                                        f"Ошибка при обработке категориальных признаков: {str(e)}")
                    return

            # Выбираем модель
            model_type = self.ui.modelComboBox.currentText()

            if model_type == "LinearRegression":
                self.model = LinearRegression()
            else:
                self.model = LinearRegression()  # По умолчанию

            # Обучаем модель
            self.model.fit(X_train, y_train)

            # Делаем предсказания
            y_pred = self.model.predict(X_test)

            # Вычисляем метрики
            metrics = self.calculate_metrics(y_test, y_pred)

            # Обновляем интерфейс
            self.update_metrics_display(metrics)
            self.update_predictions_table(y_test, y_pred)
            self.update_model_info(target, model_type, len(y_train), len(y_test))

            # Выводим информацию о модели
            self.show_model_info(metrics, len(y_train), len(y_test))

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить модель: {str(e)}")

    def update_metrics_display(self, metrics):
        """Обновление отображения метрик"""
        metrics_text = f"""
        <div style='color: #2d3748; line-height: 1.6;'>
            <div style='margin-bottom: 12px;'>
                <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Коэффициент детерминации (R²):</span><br>
                <span style='font-size: 16px; color: #2c5282; font-weight: bold;'>{metrics['R2']}</span>
            </div>

            <div style='margin-bottom: 12px;'>
                <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Средняя абсолютная процентная ошибка (MAPE):</span><br>
                <span style='font-size: 16px; color: #2c5282; font-weight: bold;'>{metrics['MAPE']}%</span>
            </div>

            <div style='margin-bottom: 12px;'>
                <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Средняя абсолютная ошибка (MAE):</span><br>
                <span style='font-size: 16px; color: #2c5282; font-weight: bold;'>{metrics['MAE']}</span>
            </div>

            <div style='margin-bottom: 12px;'>
                <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Среднеквадратичная ошибка (RMSE):</span><br>
                <span style='font-size: 16px; color: #2c5282; font-weight: bold;'>{metrics['RMSE']}</span>
            </div>
        </div>
        """
        self.ui.metricsTextEdit.setHtml(metrics_text)

    def update_predictions_table(self, y_true, y_pred):
        """Обновление таблицы с предсказаниями"""
        # Создаем DataFrame для отображения
        df_display = pd.DataFrame({
            'Фактическое': y_true.values,
            'Предсказанное': y_pred
        }).round(4)

        # Создаем модель для таблицы
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['Фактическое', 'Предсказанное'])

        # Заполняем данными (первые 100 строк)
        max_rows = min(100, len(df_display))
        for i in range(max_rows):
            row_items = [
                QStandardItem(f"{df_display.iloc[i, 0]}"),
                QStandardItem(f"{df_display.iloc[i, 1]}")
            ]
            model.appendRow(row_items)

        self.ui.predictionsTableView.setModel(model)

        # Настраиваем отображение
        header = self.ui.predictionsTableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def update_model_info(self, target, model_type, train_size, test_size):
        """Обновление информации о модели"""
        info_text = f"Модель: {model_type} | Целевая переменная: {target} | "
        info_text += f"Обучающая выборка: {train_size} строк | "
        info_text += f"Тестовая выборка: {test_size} строк"
        self.ui.modelInfoLabel.setText(info_text)

    def show_model_info(self, metrics, train_size, test_size):
        """Показ информации о построенной модели"""
        info_message = f"""
        <div style='font-size: 13px; line-height: 1.5;'>
            <h3 style='color: #1e3a5f; text-align: center; margin-top: 0; margin-bottom: 15px;'>✅ Модель успешно построена!</h3>

            <div style='background-color: #ebf8ff; padding: 10px; border-radius: 8px; margin: 8px 0;'>
                <b>Использованная модель:</b> {self.ui.modelComboBox.currentText()}<br>
                <b>Целевая переменная:</b> {self.ui.targetComboBox.currentText()}<br>
                <b>Размер обучающей выборки:</b> {train_size} строк<br>
                <b>Размер тестовой выборки:</b> {test_size} строк<br>
                <b>Тестовая выборка:</b> {self.ui.testSizeSpinBox.value()}%<br>
                <b>Random seed:</b> {self.ui.randomSeedSpinBox.value()}
            </div>

            <div style='background-color: #f0fff4; padding: 10px; border-radius: 8px; margin: 8px 0;'>
                <h4 style='color: #2f855a; margin-top: 0; margin-bottom: 6px;'>Метрики качества:</h4>
                <b style='color: #3182ce;'>R²:</b> {metrics['R2']} (чем ближе к 1, тем лучше)<br>
                <b style='color: #3182ce;'>MAPE:</b> {metrics['MAPE']}% (чем меньше, тем лучше)<br>
                <b style='color: #3182ce;'>MAE:</b> {metrics['MAE']} (чем меньше, тем лучше)<br>
                <b style='color: #3182ce;'>RMSE:</b> {metrics['RMSE']} (чем меньше, тем лучше)
            </div>
        </div>
        """

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Модель построена")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(info_message)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def closeEvent(self, event):
        """Обработка события закрытия окна"""
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    import numpy as np
    import pandas as pd


    # Создаем тестовый датасет
    def create_test_dataset():
        np.random.seed(42)
        n_samples = 1000

        # Создаем синтетические данные
        data = {
            # Числовые признаки
            'возраст': np.random.randint(18, 70, n_samples),
            'зарплата': np.random.normal(50000, 15000, n_samples),
            'стаж_работы': np.random.randint(0, 40, n_samples),
            'кредитный_скор': np.random.normal(650, 100, n_samples),
            'долг': np.random.exponential(5000, n_samples),
            'сбережения': np.random.normal(20000, 10000, n_samples),
            'расходы_в_месяц': np.random.normal(30000, 8000, n_samples),

            # Категориальные признаки
            'образование': np.random.choice(
                ['среднее', 'высшее', 'неполное_высшее', 'среднее_специальное'], n_samples),
            'семейное_положение': np.random.choice(
                ['холост/не замужем', 'женат/замужем', 'разведен/разведена', 'вдовец/вдова'],
                n_samples),
            'город': np.random.choice(
                ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань'], n_samples),
            'пол': np.random.choice(['мужской', 'женский'], n_samples),
            'наличие_детей': np.random.choice(['да', 'нет'], n_samples),

            # Логические признаки
            'ипотека': np.random.choice([True, False], n_samples),
            'автомобиль': np.random.choice([True, False], n_samples),

            # Целевые переменные для тестирования
            'стоимость_страховки': np.random.normal(30000, 8000, n_samples),
            'вероятность_дефолта': np.random.uniform(0, 1, n_samples),
            'ежемесячный_платеж': np.random.normal(15000, 4000, n_samples),
            'рейтинг_клиента': np.random.randint(1, 10, n_samples)
        }

        df = pd.DataFrame(data)

        # Добавляем целевую переменную с зависимостью от других признаков
        df['стоимость_страховки'] = (
                20000 +
                df['возраст'] * 100 +
                df['зарплата'] * 0.1 +
                df['кредитный_скор'] * 10 +
                (df['город'] == 'Москва') * 5000 +
                (df['город'] == 'Санкт-Петербург') * 3000 +
                np.random.normal(0, 2000, n_samples)
        )

        # Добавляем еще одну целевую переменную
        df['ежемесячный_платеж'] = (
                10000 +
                df['долг'] * 0.2 +
                df['сбережения'] * (-0.05) +
                df['стаж_работы'] * 200 +
                np.random.normal(0, 1500, n_samples)
        )

        return df


    # Основная функция тестирования
    def main():
        app = QApplication(sys.argv)

        # Создаем тестовый датасет
        test_df = create_test_dataset()
        print(f"Создан тестовый датасет: {test_df.shape[0]} строк, {test_df.shape[1]} столбцов")
        print("\nСтолбцы датасета:")
        print(test_df.columns.tolist())
        print("\nТипы данных:")
        print(test_df.dtypes)

        # Сохраняем датасет в CSV для тестирования
        test_file = "test_modeling_dataset.csv"
        test_df.to_csv(test_file, index=False)
        print(f"\nДатасет сохранен в файл: {test_file}")

        # Создаем и показываем окно моделирования
        window = ModelingWindow(test_file, parent=None)
        window.show()

        # Совет по выбору таргетов и фичей:
        print("\n" + "=" * 80)
        print("СОВЕТЫ ПО ИСПОЛЬЗОВАНИЮ:")
        print("=" * 80)
        print("\n📊 ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ (таргеты) для тестирования:")
        print("   1. 'стоимость_страховки' - хорошая числовая целевая переменная")
        print("   2. 'ежемесячный_платеж' - еще одна хорошая числовая целевая")
        print("   3. 'рейтинг_клиента' - дискретная числовая переменная")

        print("\n🔧 ПРИЗНАКИ (фичи) для тестирования:")
        print("   Числовые признаки:")
        print("   - 'возраст', 'зарплата', 'стаж_работы', 'кредитный_скор'")
        print("   - 'долг', 'сбережения', 'расходы_в_месяц'")

        print("\n   Категориальные признаки:")
        print("   - 'образование', 'семейное_положение', 'город', 'пол'")
        print("   - 'наличие_детей'")

        print("\n   Логические признаки:")
        print("   - 'ипотека', 'автомобиль'")

        print("\n💡 ПРИМЕРЫ КОМБИНАЦИЙ:")
        print("   1. Таргет: 'стоимость_страховки'")
        print("      Фичи: ['возраст', 'зарплата', 'кредитный_скор', 'город', 'образование']")

        print("\n   2. Таргет: 'ежемесячный_платеж'")
        print("      Фичи: ['долг', 'сбережения', 'стаж_работы', 'семейное_положение']")

        print("\n   3. Таргет: 'рейтинг_клиента'")
        print("      Фичи: ['зарплата', 'кредитный_скор', 'образование', 'город', 'ипотека']")

        print("\n⚠️  ПРЕДУПРЕЖДЕНИЯ:")
        print("   - Не выбирайте целевой переменной: 'вероятность_дефолта' (может быть проблемной)")
        print("   - Для категориальных признаков будет применен One-Hot Encoding")
        print("   - Начните с 15% тестовой выборки и random seed = 42")
        print("\n" + "=" * 80)

        sys.exit(app.exec())


    # Запускаем тестирование
    main()
