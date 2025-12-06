import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, chi2_contingency
import seaborn as sns

from PyQt6.QtWidgets import (
    QMainWindow, QHeaderView, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_percentage_error,
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
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
        self.label_encoder = LabelEncoder()
        self.categorical_features = []
        self.numerical_features = []
        self.task_type = None  # 'regression' или 'classification'
        self.is_classification = False

        # Настройка интерфейса
        self.setup_ui()

        # Загрузка данных
        self.load_data()

        # Подключение сигналов
        self.connect_signals()

    def setup_ui(self):
        """Настройка элементов интерфейса"""
        # Установка списка моделей
        self.ui.modelComboBox.addItems(["LinearRegression", "RandomForestClassifier"])
        self.setMinimumWidth(865)
        self.setMinimumHeight(812)

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
                self.df = pd.read_csv(file_path, sep=separator, encoding='utf-8')
                print(f"Загружены данные с разделителем: '{separator}'")
            else:
                # Пробуем разные разделители
                self.df = self.try_different_separators(file_path)

            # Определяем типы колонок
            self.identify_column_types()

            # Заполняем выпадающие списки
            self.populate_selection_lists()

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
                df = pd.read_csv(file_path, sep=sep, engine='python', encoding='utf-8')
                # Проверяем, что загрузилось больше одной колонки
                if len(df.columns) > 1:
                    print(f"Найден разделитель: '{sep}'")
                    return df
            except Exception as e:
                continue

        # Если ничего не помогло, пробуем загрузить с разделителем по умолчанию
        try:
            print("Использую разделитель по умолчанию ','")
            return pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            # Последняя попытка - загрузить с параметром error_bad_lines=False
            try:
                return pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip',
                                   encoding='utf-8')
            except:
                raise Exception(f"Не удалось загрузить файл {self.filename} с любым разделителем")

    def identify_column_types(self):
        """Определение типов колонок (категориальные/числовые)"""
        self.categorical_features = []
        self.numerical_features = []

        if self.df is None:
            return

        for column in self.df.columns:
            # Пропускаем полностью пустые колонки
            if self.df[column].isnull().all():
                continue

            # Проверяем на строки, которые могут быть числами
            if self.df[column].dtype in ['object', 'bool', 'category']:
                # Пробуем преобразовать в числа, если это возможно
                try:
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
        print(f"Числовые: {self.numerical_features[:10]}...")
        print(f"Категориальные: {self.categorical_features[:10]}...")

    def populate_selection_lists(self):
        """Заполнение списков выбора"""
        # Целевая переменная
        self.ui.targetComboBox.clear()
        self.ui.targetComboBox.addItems(self.df.columns.tolist())

        # Список признаков
        self.ui.featuresListWidget.clear()
        self.ui.featuresListWidget.addItems(self.df.columns.tolist())

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

    def get_ohe(self, data, categ_columns):
        """Преобразует категориальные признаки в one-hot encoding"""
        try:
            # Проверяем, что все категориальные признаки присутствуют в данных
            missing_cols = [col for col in categ_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Категориальные признаки отсутствуют в данных: {missing_cols}")

            # Преобразуем категориальные признаки
            temp_df = pd.DataFrame(
                data=self.ohe.transform(data[categ_columns]),
                columns=self.ohe.get_feature_names_out(categ_columns)
            )

            # Объединяем данные: оставляем только некатегориальные столбцы + one-hot столбцы
            non_categ_cols = [col for col in data.columns if col not in categ_columns]
            result = pd.concat([data[non_categ_cols].reset_index(drop=True),
                                temp_df.reset_index(drop=True)], axis=1)

            return result

        except Exception as e:
            print(f"Ошибка в get_ohe: {e}")
            raise

    def calculate_regression_metrics(self, fact, prediction, n_features=None):
        """Вычисление метрик для регрессии"""
        metrics = {}
        try:
            metrics['R2'] = round(r2_score(fact, prediction), 4)
        except:
            metrics['R2'] = 0.0

        # Скорректированный R² (Adjusted R²)
        try:
            n = len(fact)
            if n_features is not None and n > n_features + 1:
                adj_r2 = 1 - (1 - metrics['R2']) * (n - 1) / (n - n_features - 1)
                metrics['Adj_R2'] = round(max(adj_r2, -1), 4)  # Ограничиваем снизу -1
            else:
                metrics['Adj_R2'] = metrics['R2']
        except:
            metrics['Adj_R2'] = metrics['R2']

        try:
            metrics['MAPE'] = round(mean_absolute_percentage_error(fact, prediction) * 100, 3)
        except:
            metrics['MAPE'] = 0.0

        metrics['MAE'] = round(mean_absolute_error(fact, prediction), 4)
        metrics['RMSE'] = round(mean_squared_error(fact, prediction) ** 0.5, 4)

        # Дополнительные метрики
        metrics['MaxError'] = round(max(abs(fact - prediction)), 4)

        # Средняя абсолютная ошибка в процентах
        try:
            metrics['MAE_percent'] = round(100 * metrics['MAE'] / np.mean(np.abs(fact)), 2)
        except:
            metrics['MAE_percent'] = 0.0

        return metrics

    def calculate_classification_metrics(self, y_true, y_pred):
        """Вычисление метрик для классификации"""
        metrics = {}

        try:
            metrics['Accuracy'] = round(accuracy_score(y_true, y_pred), 4)
            metrics['Precision'] = round(
                precision_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            metrics['Recall'] = round(
                recall_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            metrics['F1-Score'] = round(
                f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)

            # Для бинарной классификации считаем дополнительные метрики
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['ROC-AUC'] = round(roc_auc_score(y_true, y_pred), 4)
                except:
                    metrics['ROC-AUC'] = np.nan
            else:
                metrics['ROC-AUC'] = np.nan

            # Матрица ошибок
            cm = confusion_matrix(y_true, y_pred)
            metrics['ConfusionMatrix'] = cm

            # Отчет классификации
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['ClassificationReport'] = report

        except Exception as e:
            print(f"Ошибка при вычислении метрик классификации: {e}")
            metrics = {'Error': str(e)}

        return metrics

    def build_model(self):
        """Построение модели"""
        try:
            # Получаем выбранные параметры
            target = self.ui.targetComboBox.currentText()
            test_size = self.ui.testSizeSpinBox.value() / 100
            random_state = self.ui.randomSeedSpinBox.value()
            model_type = self.ui.modelComboBox.currentText()

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

            # Определяем тип задачи на основе целевой переменной
            target_unique = self.df[target].nunique()
            # Если уникальных значений <= 10, считаем это классификацией
            self.is_classification = target_unique <= 10 and target_unique >= 2

            # Проверяем, что выбранная модель подходит для типа задачи
            if self.is_classification and 'Regression' in model_type:
                QMessageBox.warning(self, "Ошибка",
                                    f"Модель {model_type} предназначена для регрессии, "
                                    f"а целевая переменная '{target}' имеет {target_unique} "
                                    f"уникальных значений (классификация).\n"
                                    f"Выберите RandomForestClassifier.")
                return
            elif not self.is_classification and 'Classifier' in model_type:
                QMessageBox.warning(self, "Ошибка",
                                    f"Модель {model_type} предназначена для классификации, "
                                    f"а целевая переменная '{target}' имеет {target_unique} "
                                    f"уникальных значений (регрессия).\n"
                                    f"Выберите LinearRegression.")
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

            # Для классификации проверяем, что есть хотя бы 2 класса
            if self.is_classification:
                unique_classes = df_clean[target].nunique()
                if unique_classes < 2:
                    QMessageBox.warning(self, "Ошибка",
                                        f"Для классификации нужно минимум 2 класса. Найдено: {unique_classes}")
                    return

            # Проверяем, можно ли использовать стратификацию для классификации
            use_stratify = False
            if self.is_classification:
                # Проверяем, что в каждом классе минимум 2 элемента для стратификации
                class_counts = df_clean[target].value_counts()
                min_class_size = class_counts.min()

                if min_class_size >= 2:
                    use_stratify = True
                    print(f"Используем стратификацию. Размеры классов: {dict(class_counts)}")
                else:
                    print(
                        f"Не используем стратификацию. Минимальный размер класса: {min_class_size}")
                    QMessageBox.warning(self, "Предупреждение",
                                        f"Самый малочисленный класс содержит только {min_class_size} элемент(а).\n"
                                        f"Стратификация не будет использована. Классы: {dict(class_counts)}")

            try:
                # Разделяем данные
                if use_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        df_clean[selected_features],
                        df_clean[target],
                        test_size=test_size,
                        random_state=random_state,
                        stratify=df_clean[target]
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        df_clean[selected_features],
                        df_clean[target],
                        test_size=test_size,
                        random_state=random_state
                    )
            except Exception as e:
                # Если возникает ошибка при разделении, пробуем без стратификации
                print(f"Ошибка при разделении данных: {e}. Пробуем без стратификации...")
                X_train, X_test, y_train, y_test = train_test_split(
                    df_clean[selected_features],
                    df_clean[target],
                    test_size=test_size,
                    random_state=random_state
                )

            print(f"Обучающая выборка: {len(X_train)} записей")
            print(f"Тестовая выборка: {len(X_test)} записей")
            print(f"Тип задачи: {'Классификация' if self.is_classification else 'Регрессия'}")
            print(f"Уникальных классов: {df_clean[target].nunique()}")
            if self.is_classification:
                print(
                    f"Распределение классов в обучающей выборке: {y_train.value_counts().to_dict()}")
                print(
                    f"Распределение классов в тестовой выборке: {y_test.value_counts().to_dict()}")

            # Обрабатываем категориальные признаки
            if categorical_features:
                try:
                    # Объединяем тренировочные и тестовые данные для корректного обучения
                    combined_categ_data = pd.concat([
                        X_train[categorical_features],
                        X_test[categorical_features]
                    ], axis=0)

                    # Обучаем кодировщик на объединенных данных
                    self.ohe.fit(combined_categ_data)

                    # Преобразуем тренировочные и тестовые данные
                    X_train = self.get_ohe(X_train, categorical_features)
                    X_test = self.get_ohe(X_test, categorical_features)

                    # Убеждаемся, что X_test имеет те же столбцы, что и X_train
                    missing_cols = set(X_train.columns) - set(X_test.columns)
                    for col in missing_cols:
                        X_test[col] = 0

                    # Упорядочиваем столбцы в X_test так же, как в X_train
                    X_test = X_test[X_train.columns]

                except Exception as e:
                    QMessageBox.warning(self, "Предупреждение",
                                        f"Ошибка при обработке категориальных признаков: {str(e)}")
                    return

            # Создаем и обучаем модель
            if model_type == "LinearRegression":
                self.model = LinearRegression()
            elif model_type == "RandomForestClassifier":
                # Для RandomForest используем 100 деревьев по умолчанию
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    max_depth=None,  # Неограниченная глубина
                    min_samples_split=2,
                    min_samples_leaf=1
                )
            else:
                QMessageBox.warning(self, "Ошибка", f"Неизвестный тип модели: {model_type}")
                return

            print(f"Обучаем модель: {model_type}")
            print(f"Размер X_train: {X_train.shape}")
            print(f"Размер y_train: {y_train.shape}")

            # Для классификации кодируем целевую переменную если она строковая
            if self.is_classification and y_train.dtype == 'object':
                y_train_encoded = self.label_encoder.fit_transform(y_train)
                y_test_encoded = self.label_encoder.transform(y_test)
                self.model.fit(X_train, y_train_encoded)
                y_pred = self.model.predict(X_test)
                y_pred_original = self.label_encoder.inverse_transform(y_pred)
                y_test_original = self.label_encoder.inverse_transform(y_test_encoded)
            else:
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                y_pred_original = y_pred
                y_test_original = y_test

            # Вычисляем метрики
            if self.is_classification:
                if y_train.dtype == 'object':
                    metrics = self.calculate_classification_metrics(y_test_encoded, y_pred)
                else:
                    metrics = self.calculate_classification_metrics(y_test, y_pred)
            else:
                n_features = X_train.shape[1]
                metrics = self.calculate_regression_metrics(y_test, y_pred, n_features)

            # Обновляем интерфейс
            if self.is_classification:
                self.update_classification_metrics_display(metrics)
                headers = ['Фактическое', 'Предсказанное']
            else:
                self.update_regression_metrics_display(metrics)
                headers = ['Фактическое', 'Предсказанное']

            self.update_predictions_table(y_test_original, y_pred_original, headers)
            self.update_model_info(target, model_type, len(y_train), len(y_test),
                                   self.is_classification, metrics)

            # Выводим информацию о модели
            self.show_model_info(metrics, len(y_train), len(y_test),
                                 self.is_classification, model_type, target)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить модель: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_predictions_table(self, y_true, y_pred, headers):
        """Обновление таблицы с предсказаниями"""
        # Создаем DataFrame для отображения
        df_display = pd.DataFrame({
            headers[0]: y_true.values if hasattr(y_true, 'values') else y_true,
            headers[1]: y_pred
        })

        # Округляем числовые значения
        if not self.is_classification:
            df_display = df_display.round(4)

        # Создаем модель для таблицы
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(headers)

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
        header_view = self.ui.predictionsTableView.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def update_model_info(self, target, model_type, train_size, test_size,
                          is_classification, metrics):
        """Обновление информации о модели"""
        task_type = "Классификация" if is_classification else "Регрессия"

        if is_classification:
            main_metric = f"Accuracy: {metrics.get('Accuracy', 'N/A')}"
        else:
            main_metric = f"R²: {metrics.get('R2', 'N/A')}"

        info_text = f"Модель: {model_type} | Задача: {task_type} | Целевая: {target} | "
        info_text += f"Обучающая: {train_size} | Тестовая: {test_size} | {main_metric}"
        self.ui.modelInfoLabel.setText(info_text)

    def update_regression_metrics_display(self, metrics):
        """Обновление отображения метрик для регрессии с пояснениями"""
        metrics_text = f"""
        <div style='color: #2d3748; line-height: 1.6;'>
            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #3182ce;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Коэффициент детерминации (R²):</span>
                    <span style='font-size: 16px; color: #2c5282; font-weight: bold;'>{metrics.get('R2', 'N/A')}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Показывает, какая доля дисперсии зависимой переменной объясняется моделью.<br>
                    <b>Интерпретация:</b> от 0 до 1. Чем ближе к 1, тем лучше.
                </div>
            </div>

            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #3182ce;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Скорректированный R² (Adj R²):</span>
                    <span style='font-size: 16px; color: #2c5282; font-weight: bold;'>{metrics.get('Adj_R2', 'N/A')}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    R² с поправкой на количество признаков. Учитывает сложность модели.<br>
                    <b>Интерпретация:</b> Более честная оценка, чем R², особенно при многих признаках.
                </div>
            </div>

            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #e53e3e;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Средняя абсолютная процентная ошибка (MAPE):</span>
                    <span style='font-size: 16px; color: #c53030; font-weight: bold;'>{metrics.get('MAPE', 'N/A')}%</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Средняя абсолютная ошибка в процентах от фактических значений.<br>
                    <b>Интерпретация:</b> Чем меньше, тем лучше. Например, MAPE=10% означает среднюю ошибку 10%.
                </div>
            </div>

            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #e53e3e;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Средняя абсолютная ошибка (MAE):</span>
                    <div>
                        <span style='font-size: 16px; color: #c53030; font-weight: bold;'>{metrics.get('MAE', 'N/A')}</span>
                        <span style='font-size: 12px; color: #a0aec0; margin-left: 5px;'>({metrics.get('MAE_percent', 'N/A')}% от среднего)</span>
                    </div>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Среднее абсолютное значение разницы между предсказанными и фактическими значениями.<br>
                    <b>Интерпретация:</b> Менее чувствительна к выбросам, чем RMSE. Чем меньше, тем лучше.
                </div>
            </div>

            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #e53e3e;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Среднеквадратичная ошибка (RMSE):</span>
                    <span style='font-size: 16px; color: #c53030; font-weight: bold;'>{metrics.get('RMSE', 'N/A')}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Корень из среднего квадрата разностей между предсказанными и фактическими значениями.<br>
                    <b>Интерпретация:</b> Более чувствительна к большим ошибкам (выбросам). Чем меньше, тем лучше.
                </div>
            </div>
        </div>
        """
        self.ui.metricsTextEdit.setHtml(metrics_text)

    def update_classification_metrics_display(self, metrics):
        """Обновление отображения метрик для классификации с пояснениями"""
        # Формируем текст для ROC-AUC
        roc_auc_text = ""
        if 'ROC-AUC' in metrics and not pd.isna(metrics['ROC-AUC']):
            roc_auc_value = metrics.get('ROC-AUC', 'N/A')
            roc_auc_interpretation = self.get_roc_auc_interpretation(roc_auc_value)
            roc_auc_text = f"""
            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #38a169;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>ROC-AUC (Площадь под ROC-кривой):</span>
                    <span style='font-size: 16px; color: #2f855a; font-weight: bold;'>{roc_auc_value}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Показывает способность модели различать классы.<br>
                    <b>Интерпретация:</b> {roc_auc_interpretation}
                </div>
            </div>
            """

        metrics_text = f"""
        <div style='color: #2d3748; line-height: 1.6;'>
            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #38a169;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Точность (Accuracy):</span>
                    <span style='font-size: 16px; color: #2f855a; font-weight: bold;'>{metrics.get('Accuracy', 'N/A')}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Доля правильных предсказаний среди всех предсказаний.<br>
                    <b>Интерпретация:</b> от 0 до 1. Чем ближе к 1, тем лучше.
                </div>
            </div>

            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #38a169;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Precision (Точность):</span>
                    <span style='font-size: 16px; color: #2f855a; font-weight: bold;'>{metrics.get('Precision', 'N/A')}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Из всех предсказанных положительных случаев, сколько действительно положительных.<br>
                    <b>Интерпретация:</b> Чем выше, тем меньше ложных срабатываний.
                </div>
            </div>

            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #38a169;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>Recall (Полнота):</span>
                    <span style='font-size: 16px; color: #2f855a; font-weight: bold;'>{metrics.get('Recall', 'N/A')}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Из всех реальных положительных случаев, сколько правильно предсказано.<br>
                    <b>Интерпретация:</b> Чем выше, тем меньше пропущенных положительных случаев.
                </div>
            </div>

            <div style='margin-bottom: 16px; background-color: #f7fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #38a169;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                    <span style='font-weight: bold; color: #3182ce; font-size: 13px;'>F1-Score (F-мера):</span>
                    <span style='font-size: 16px; color: #2f855a; font-weight: bold;'>{metrics.get('F1-Score', 'N/A')}</span>
                </div>
                <div style='font-size: 11px; color: #718096;'>
                    Гармоническое среднее между Precision и Recall.<br>
                    <b>Интерпретация:</b> Баланс между точностью и полнотой. Особенно важен при несбалансированных классах.
                </div>
            </div>

            {roc_auc_text}
        </div>
        """
        self.ui.metricsTextEdit.setHtml(metrics_text)

    def get_roc_auc_interpretation(self, value):
        """Возвращает текстовую интерпретацию значения ROC-AUC"""
        if value == 'N/A' or pd.isna(value):
            return "Не вычислено (только для бинарной классификации)"

        try:
            value = float(value)
            if value >= 0.9:
                return "Отлично! Очень высокая разделяющая способность"
            elif value >= 0.8:
                return "Хорошо. Хорошая разделяющая способность"
            elif value >= 0.7:
                return "Удовлетворительно. Приемлемая разделяющая способность"
            elif value >= 0.6:
                return "Слабо. Разделяющая способность ниже среднего"
            elif value >= 0.5:
                return "Плохо. Модель почти не лучше случайного угадывания"
            else:
                return "Очень плохо. Модель работает хуже случайного угадывания"
        except:
            return "Не удалось интерпретировать значение"

    def show_model_info(self, metrics, train_size, test_size,
                        is_classification, model_type, target):
        """Показ информации о построенной модели"""
        if is_classification:
            # Формируем текст для ROC-AUC в сообщении
            roc_auc_info = ""
            if 'ROC-AUC' in metrics and not pd.isna(metrics['ROC-AUC']):
                roc_auc_value = metrics.get('ROC-AUC', 'N/A')
                roc_auc_interpretation = self.get_roc_auc_interpretation(roc_auc_value)
                roc_auc_info = f"<b style='color: #3182ce;'>ROC-AUC:</b> {roc_auc_value} - {roc_auc_interpretation}<br>"

            metrics_text = f"""
            <div style='background-color: #f0fff4; padding: 10px; border-radius: 8px; margin: 8px 0;'>
                <h4 style='color: #2f855a; margin-top: 0; margin-bottom: 6px;'>Метрики качества классификации:</h4>
                <b style='color: #3182ce;'>Accuracy (Точность):</b> {metrics.get('Accuracy', 'N/A')} - доля правильных предсказаний<br>
                <b style='color: #3182ce;'>Precision (Точность положительных):</b> {metrics.get('Precision', 'N/A')} - точность определения положительных классов<br>
                <b style='color: #3182ce;'>Recall (Полнота):</b> {metrics.get('Recall', 'N/A')} - способность находить положительные классы<br>
                <b style='color: #3182ce;'>F1-Score (F-мера):</b> {metrics.get('F1-Score', 'N/A')} - баланс между точностью и полнотой<br>
                {roc_auc_info}
            </div>
            """
        else:
            metrics_text = f"""
            <div style='background-color: #f0fff4; padding: 10px; border-radius: 8px; margin: 8px 0;'>
                <h4 style='color: #2f855a; margin-top: 0; margin-bottom: 6px;'>Метрики качества регрессии:</h4>
                <b style='color: #3182ce;'>R² (Коэффициент детерминации):</b> {metrics.get('R2', 'N/A')} - доля объясненной дисперсии<br>
                <b style='color: #3182ce;'>Adj R² (Скорректированный R²):</b> {metrics.get('Adj_R2', 'N/A')} - R² с учетом сложности модели<br>
                <b style='color: #3182ce;'>MAPE (Средняя абсолютная процентная ошибка):</b> {metrics.get('MAPE', 'N/A')}% - средняя ошибка в процентах<br>
                <b style='color: #3182ce;'>MAE (Средняя абсолютная ошибка):</b> {metrics.get('MAE', 'N/A')} ({metrics.get('MAE_percent', 'N/A')}% от среднего)<br>
                <b style='color: #3182ce;'>RMSE (Среднеквадратичная ошибка):</b> {metrics.get('RMSE', 'N/A')} - чувствительна к выбросам
            </div>
            """

        info_message = f"""
        <div style='font-size: 13px; line-height: 1.5;'>
            <h3 style='color: #1e3a5f; text-align: center; margin-top: 0; margin-bottom: 15px;'>Модель успешно построена!</h3>

            <div style='background-color: #ebf8ff; padding: 10px; border-radius: 8px; margin: 8px 0;'>
                <b>Использованная модель:</b> {model_type}<br>
                <b>Тип задачи:</b> {"Классификация" if is_classification else "Регрессия"}<br>
                <b>Целевая переменная:</b> {target}<br>
                <b>Размер обучающей выборки:</b> {train_size} строк<br>
                <b>Размер тестовой выборки:</b> {test_size} строк<br>
                <b>Тестовая выборка:</b> {self.ui.testSizeSpinBox.value()}%<br>
                <b>Random seed:</b> {self.ui.randomSeedSpinBox.value()}
            </div>

            {metrics_text}

            <div style='background-color: #fed7d7; padding: 8px; border-radius: 6px; margin: 8px 0; font-size: 12px;'>
                <b>Совет:</b> {"Для улучшения точности классификации попробуйте:" if is_classification else "Для улучшения качества регрессии попробуйте:"}<br>
                {"• Добавить больше признаков" if is_classification else "• Добавить больше признаков"}<br>
                {"• Увеличить количество данных" if is_classification else "• Увеличить количество данных"}<br>
                {"• Убрать несбалансированные классы" if is_classification else "• Убрать выбросы в данных"}
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
