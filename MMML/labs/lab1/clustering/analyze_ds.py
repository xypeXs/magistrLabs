import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
import json

warnings.filterwarnings('ignore')

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.rcParams['font.size'] = 10


class DatasetCharacteristicsAnalyzer:
    """
    Класс для комплексного анализа характеристик датасета
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.analysis_results = {}

    def load_dataset(self):
        """Загрузка и объединение данных датасета"""
        print("Загрузка датасета...")
        all_data = []

        for activity_folder in os.listdir(self.dataset_path):
            activity_path = os.path.join(self.dataset_path, activity_folder)

            if os.path.isdir(activity_path):
                for file in os.listdir(activity_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(activity_path, file)

                        try:
                            # Читаем данные акселерометра
                            accel_data = pd.read_csv(file_path, header=None,
                                                     names=['accel_x', 'accel_y', 'accel_z'],
                                                     dtype=float, delim_whitespace=True)

                            # Парсим мета-информацию из имени файла
                            filename_parts = file.replace('.txt', '').split('-')
                            if len(filename_parts) >= 9:
                                activity = filename_parts[7]
                                volunteer = filename_parts[8]

                                accel_data['activity'] = activity
                                accel_data['volunteer'] = volunteer
                                accel_data['file_source'] = file
                                accel_data['sample_id'] = range(len(accel_data))

                                all_data.append(accel_data)

                        except Exception as e:
                            continue  # Пропускаем проблемные файлы

        if all_data:
            self.df = pd.concat(all_data, ignore_index=True)
            print(f"Успешно загружено {len(self.df)} записей")
            print(f"Колонки: {list(self.df.columns)}")
        else:
            raise ValueError("Не удалось загрузить данные")

    def analyze_size_and_dimensions(self):
        """Анализ размера и размерности датасета"""
        print("Анализ размера и размерности...")

        size_info = {
            'total_objects': len(self.df),
            'total_features': len(self.df.columns),
            'numerical_features': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(self.df.select_dtypes(include=['object']).columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 ** 2,
            'data_density': self._calculate_data_density()
        }

        # Анализ по активностям
        activity_stats = {
            'unique_activities': self.df['activity'].nunique(),
            'activities_list': list(self.df['activity'].unique()),
            'records_per_activity': self.df['activity'].value_counts().to_dict(),
            'min_records_per_activity': self.df['activity'].value_counts().min(),
            'max_records_per_activity': self.df['activity'].value_counts().max(),
            'mean_records_per_activity': self.df['activity'].value_counts().mean()
        }

        # Анализ по волонтерам
        volunteer_stats = {
            'unique_volunteers': self.df['volunteer'].nunique(),
            'volunteers_list': list(self.df['volunteer'].unique()),
            'records_per_volunteer': self.df['volunteer'].value_counts().to_dict(),
            'activities_per_volunteer': self.df.groupby('volunteer')['activity'].nunique().to_dict(),
            'min_records_per_volunteer': self.df['volunteer'].value_counts().min(),
            'max_records_per_volunteer': self.df['volunteer'].value_counts().max()
        }

        self.analysis_results['size_dimensions'] = {
            'dataset_size': size_info,
            'activity_analysis': activity_stats,
            'volunteer_analysis': volunteer_stats
        }

    def _calculate_data_density(self):
        """Расчет плотности данных (доля заполненных значений)"""
        total_cells = self.df.size
        non_null_cells = self.df.count().sum()
        return non_null_cells / total_cells

    def analyze_data_types(self):
        """Анализ типов данных"""
        print("Анализ типов данных...")

        data_types = {
            'dtypes_summary': {str(k): str(v) for k, v in self.df.dtypes.to_dict().items()},
            'numerical_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'boolean_columns': list(self.df.select_dtypes(include=['bool']).columns)
        }

        # Детальный анализ числовых данных
        numerical_analysis = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            numerical_analysis[col] = {
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'median': float(self.df[col].median()),
                'unique_values': int(self.df[col].nunique()),
                'data_type': str(self.df[col].dtype)
            }

        # Анализ категориальных данных
        categorical_analysis = {}
        for col in self.df.select_dtypes(include=['object']).columns:
            value_counts = self.df[col].value_counts()
            most_frequent = value_counts.index[0] if len(value_counts) > 0 else None

            categorical_analysis[col] = {
                'unique_values': int(self.df[col].nunique()),
                'most_frequent': str(most_frequent) if most_frequent else None,
                'frequency_top5': value_counts.head(5).to_dict()
            }

        self.analysis_results['data_types'] = {
            'type_overview': data_types,
            'numerical_details': numerical_analysis,
            'categorical_details': categorical_analysis
        }

    def analyze_data_sparsity(self):
        """Анализ разреженности данных"""
        print("Анализ разреженности...")

        null_values_by_column = self.df.isnull().sum().to_dict()
        sparsity_info = {
            'overall_density': self._calculate_data_density(),
            'null_values_total': int(self.df.isnull().sum().sum()),
            'null_values_by_column': null_values_by_column,
            'completely_null_columns': list(self.df.columns[self.df.isnull().all()]),
            'sparse_columns': []
        }

        # Анализ разреженности по столбцам
        for col in self.df.columns:
            null_ratio = self.df[col].isnull().mean()
            if null_ratio > 0.5:  # Столбцы с более чем 50% пропусков
                sparsity_info['sparse_columns'].append({
                    'column': col,
                    'null_ratio': float(null_ratio),
                    'null_count': int(self.df[col].isnull().sum())
                })

        # Анализ дубликатов
        exact_duplicates = self.df.duplicated().sum()
        duplicate_info = {
            'exact_duplicates': int(exact_duplicates),
            'duplicate_ratio': float(exact_duplicates / len(self.df)),
            'activity_duplicates': int(
                self.df.duplicated(subset=['activity', 'volunteer', 'accel_x', 'accel_y', 'accel_z']).sum())
        }

        self.analysis_results['sparsity'] = {
            'null_analysis': sparsity_info,
            'duplicate_analysis': duplicate_info
        }

    def analyze_data_resolution(self):
        """Анализ разрешения и детализации данных"""
        print("Анализ разрешения данных...")

        # Анализ временного разрешения (предполагаемая частота дискретизации)
        acceleration_stats = {}
        for col in ['accel_x', 'accel_y', 'accel_z']:
            if col in self.df.columns:
                col_data = self.df[col]
                acceleration_stats[col] = {
                    'value_range': [float(col_data.min()), float(col_data.max())],
                    'precision_decimal_places': self._estimate_precision(col_data),
                    'unique_value_ratio': float(col_data.nunique() / len(col_data)),
                    'measurement_units': 'raw accelerometer units',
                    'data_range': float(col_data.max() - col_data.min())
                }

        # Анализ детализации по активностям
        activity_resolution = {}
        for activity in self.df['activity'].unique():
            activity_data = self.df[self.df['activity'] == activity]
            activity_resolution[activity] = {
                'samples_count': len(activity_data),
                'unique_volunteers': activity_data['volunteer'].nunique(),
                'duration_estimate_samples': len(activity_data),
                'data_completeness': float(1.0 - activity_data.isnull().mean().mean())
            }

        resolution_info = {
            'sensor_resolution': acceleration_stats,
            'temporal_resolution_estimated_hz': 32,
            'spatial_resolution': '3-axis accelerometer',
            'activity_resolution': activity_resolution,
            'volunteer_coverage': self.df.groupby('volunteer')['activity'].nunique().describe().to_dict()
        }

        self.analysis_results['resolution'] = resolution_info

    def _estimate_precision(self, data):
        """Оценка точности числовых данных"""
        if data.dtype in [np.float64, np.float32]:
            try:
                # Берем небольшую выборку для анализа
                sample_data = data.head(1000)
                decimal_places = sample_data.apply(
                    lambda x: len(str(float(x)).split('.')[1]) if '.' in str(float(x)) else 0
                )
                if len(decimal_places) > 0:
                    return int(decimal_places.mode().iloc[0]) if not decimal_places.mode().empty else 0
            except:
                pass
        return 0

    def analyze_data_quality(self):
        """Анализ качества данных"""
        print("Анализ качества данных...")

        # 1. Анализ пропущенных значений
        missing_analysis = {
            'total_missing': int(self.df.isnull().sum().sum()),
            'missing_by_column': self.df.isnull().sum().to_dict(),
            'missing_patterns': self._analyze_missing_patterns(),
            'completeness_score': float(1 - (self.df.isnull().sum().sum() / self.df.size))
        }

        # 2. Анализ аномалий и выбросов
        outlier_analysis = self._analyze_outliers()

        # 3. Анализ согласованности данных
        consistency_analysis = self._analyze_consistency()

        # 4. Анализ ошибок в категориальных данных
        categorical_errors = self._analyze_categorical_errors()

        quality_metrics = {
            'missing_values': missing_analysis,
            'outliers': outlier_analysis,
            'consistency': consistency_analysis,
            'categorical_quality': categorical_errors,
            'overall_quality_score': self._calculate_quality_score(missing_analysis, outlier_analysis)
        }

        self.analysis_results['quality'] = quality_metrics

    def _analyze_missing_patterns(self):
        """Анализ паттернов пропущенных значений"""
        patterns = {
            'completely_missing_columns': list(self.df.columns[self.df.isnull().all()]),
            'partially_missing_columns': [col for col in self.df.columns if
                                          self.df[col].isnull().any() and not self.df[col].isnull().all()],
            'missing_by_activity': self.df.groupby('activity').apply(lambda x: x.isnull().sum().sum()).to_dict(),
        }
        return patterns

    def _analyze_outliers(self):
        """Анализ выбросов с использованием IQR метода"""
        outlier_info = {}

        for col in self.df.select_dtypes(include=[np.number]).columns:
            try:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]

                outlier_info[col] = {
                    'outlier_count': len(outliers),
                    'outlier_ratio': len(outliers) / len(self.df),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'min_value': float(self.df[col].min()),
                    'max_value': float(self.df[col].max())
                }
            except Exception as e:
                outlier_info[col] = {'error': str(e)}

        return outlier_info

    def _analyze_consistency(self):
        """Анализ согласованности данных"""
        consistency_checks = {}

        # Проверка физической согласованности ускорений
        if all(col in self.df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
            acceleration_magnitude = np.sqrt(
                self.df['accel_x'] ** 2 + self.df['accel_y'] ** 2 + self.df['accel_z'] ** 2)
            physically_implausible = acceleration_magnitude > 1000

            consistency_checks['acceleration_consistency'] = {
                'implausible_values': int(physically_implausible.sum()),
                'implausible_ratio': float(physically_implausible.mean()),
                'magnitude_range': [float(acceleration_magnitude.min()), float(acceleration_magnitude.max())]
            }

        # Проверка согласованности категориальных данных
        if 'activity' in self.df.columns and 'volunteer' in self.df.columns:
            activity_volunteer_consistency = self.df.groupby(['activity', 'volunteer']).size()
            consistency_checks['activity_volunteer_distribution'] = {
                'unique_combinations': len(activity_volunteer_consistency),
                'min_samples_per_combination': int(activity_volunteer_consistency.min()),
                'max_samples_per_combination': int(activity_volunteer_consistency.max())
            }

        return consistency_checks

    def _analyze_categorical_errors(self):
        """Анализ ошибок в категориальных данных"""
        errors = {}

        for col in self.df.select_dtypes(include=['object']).columns:
            # Поиск потенциальных опечаток (очень редких значений)
            value_counts = self.df[col].value_counts()
            rare_values = value_counts[value_counts == 1]

            errors[col] = {
                'total_unique': len(value_counts),
                'rare_values_count': len(rare_values),
                'rare_values_list': list(rare_values.index),
                'inconsistent_capitalization': self._check_capitalization(self.df[col])
            }

        return errors

    def _check_capitalization(self, series):
        """Проверка несогласованности в капитализации"""
        if series.dtype == 'object':
            unique_values = series.unique()
            lower_values = [str(x).lower() for x in unique_values]
            return len(set(lower_values)) != len(unique_values)
        return False

    def _calculate_quality_score(self, missing_analysis, outlier_analysis):
        """Расчет общего показателя качества данных"""
        completeness = missing_analysis['completeness_score']

        # Оценка на основе выбросов (меньше выбросов = лучше качество)
        outlier_ratios = []
        for info in outlier_analysis.values():
            if 'outlier_ratio' in info:
                outlier_ratios.append(info['outlier_ratio'])

        if outlier_ratios:
            outlier_penalty = sum(outlier_ratios) / len(outlier_ratios)
            outlier_score = 1 - outlier_penalty
        else:
            outlier_score = 1.0

        return (completeness + outlier_score) / 2

    def analyze_usage_purpose(self):
        """Анализ цели использования датасета"""
        print("Анализ цели использования...")

        purpose_analysis = {
            'primary_purpose': 'Activity Recognition - Classification of Human Activities from Accelerometer Data',
            'suitable_tasks': [
                'Multiclass classification of human activities',
                'Time series analysis of sensor data',
                'Behavioral pattern recognition',
                'Wearable sensor data processing',
                'Human Activity Recognition (HAR) research'
            ],
            'data_type': 'Multivariate time series',
            'problem_type': 'Supervised learning - Classification',
            'target_variable': 'activity',
            'features_type': 'Numerical sensor readings + categorical metadata',
            'dataset_scale': 'Medium-sized research dataset',
            'recommended_models': [
                'Random Forest', 'Gradient Boosting', 'SVM',
                'LSTM networks', '1D CNNs', 'Ensemble methods'
            ],
            'validation_strategy': 'Stratified cross-validation by activity and volunteer',
            'potential_applications': [
                'Healthcare monitoring',
                'Elderly care systems',
                'Sports activity tracking',
                'Behavioral studies',
                'Smart home applications'
            ]
        }

        # Анализ пригодности для различных задач
        suitability_scores = {
            'classification': 0.95,
            'clustering': 0.70,
            'anomaly_detection': 0.80,
            'time_series_forecasting': 0.60,
            'transfer_learning': 0.85
        }

        purpose_analysis['suitability_scores'] = suitability_scores

        self.analysis_results['usage_purpose'] = purpose_analysis

    def generate_comprehensive_report(self):
        """Генерация комплексного отчета"""
        report = []
        report.append("=" * 100)
        report.append("КОМПЛЕКСНЫЙ АНАЛИЗ ХАРАКТЕРИСТИК ДАТАСЕТА ADL RECOGNITION")
        report.append("=" * 100)

        # 1. Размер и размерность
        size_info = self.analysis_results['size_dimensions']['dataset_size']
        activity_info = self.analysis_results['size_dimensions']['activity_analysis']
        volunteer_info = self.analysis_results['size_dimensions']['volunteer_analysis']

        report.append("\n1. РАЗМЕР И РАЗМЕРНОСТЬ ДАТАСЕТА:")
        report.append(f"   • Общее количество объектов: {size_info['total_objects']:,}")
        report.append(f"   • Количество признаков: {size_info['total_features']}")
        report.append(f"   • Числовые признаки: {size_info['numerical_features']}")
        report.append(f"   • Категориальные признаки: {size_info['categorical_features']}")
        report.append(f"   • Используемая память: {size_info['memory_usage_mb']:.2f} MB")
        report.append(f"   • Плотность данных: {size_info['data_density']:.3%}")

        report.append(f"\n   • Уникальные активности: {activity_info['unique_activities']}")
        report.append(f"   • Уникальные волонтеры: {volunteer_info['unique_volunteers']}")
        report.append(
            f"   • Записей на активность: {activity_info['min_records_per_activity']:,} - {activity_info['max_records_per_activity']:,}")

        # 2. Типы данных
        type_info = self.analysis_results['data_types']['type_overview']
        report.append("\n2. ТИПЫ ДАННЫХ:")
        report.append("   • Числовые столбцы: " + ", ".join(type_info['numerical_columns']))
        report.append("   • Категориальные столбцы: " + ", ".join(type_info['categorical_columns']))

        # 3. Разреженность
        sparsity_info = self.analysis_results['sparsity']['null_analysis']
        duplicate_info = self.analysis_results['sparsity']['duplicate_analysis']
        report.append("\n3. РАЗРЕЖЕННОСТЬ ДАННЫХ:")
        report.append(f"   • Всего пропущенных значений: {sparsity_info['null_values_total']}")
        report.append(f"   • Доля заполненных значений: {sparsity_info['overall_density']:.3%}")
        report.append(f"   • Дубликаты: {duplicate_info['exact_duplicates']} ({duplicate_info['duplicate_ratio']:.3%})")

        # 4. Разрешение
        resolution_info = self.analysis_results['resolution']
        report.append("\n4. РАЗРЕШЕНИЕ И ДЕТАЛИЗАЦИЯ:")
        report.append(f"   • Пространственное разрешение: {resolution_info['spatial_resolution']}")
        report.append(
            f"   • Предполагаемая временная частота: {resolution_info['temporal_resolution_estimated_hz']} Гц")

        # 5. Качество данных
        quality_info = self.analysis_results['quality']
        report.append("\n5. КАЧЕСТВО ДАННЫХ:")
        report.append(f"   • Общий показатель качества: {quality_info['overall_quality_score']:.3f}")
        report.append(f"   • Полнота данных: {quality_info['missing_values']['completeness_score']:.3%}")

        # Анализ выбросов
        outlier_total = sum(info.get('outlier_count', 0) for info in quality_info['outliers'].values())
        report.append(f"   • Всего выбросов: {outlier_total:,}")

        # 6. Цель использования
        usage_info = self.analysis_results['usage_purpose']
        report.append("\n6. ЦЕЛЬ ИСПОЛЬЗОВАНИЯ:")
        report.append(f"   • Основное назначение: {usage_info['primary_purpose']}")
        report.append(f"   • Тип проблемы: {usage_info['problem_type']}")
        report.append(f"   • Целевая переменная: {usage_info['target_variable']}")
        report.append(f"   • Пригодность для классификации: {usage_info['suitability_scores']['classification']:.1%}")

        report.append("\n" + "=" * 100)

        return '\n'.join(report)

    def create_detailed_visualizations(self, output_dir='dataset_characteristics'):
        """Создание детальных визуализаций характеристик"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # 1. Распределение активностей
            plt.figure(figsize=(14, 8))

            # Распределение по активностям
            plt.subplot(2, 2, 1)
            activity_counts = self.df['activity'].value_counts()
            activity_counts.plot(kind='bar')
            plt.title('Распределение записей по активностям')
            plt.xticks(rotation=45)
            plt.ylabel('Количество записей')

            # Распределение по волонтерам (топ-15)
            plt.subplot(2, 2, 2)
            volunteer_counts = self.df['volunteer'].value_counts().head(15)
            volunteer_counts.plot(kind='bar')
            plt.title('Топ-15 волонтеров по количеству записей')
            plt.xticks(rotation=45)
            plt.ylabel('Количество записей')

            # Типы данных
            plt.subplot(2, 2, 3)
            type_counts = {
                'Numerical': len(self.df.select_dtypes(include=[np.number]).columns),
                'Categorical': len(self.df.select_dtypes(include=['object']).columns)
            }
            plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            plt.title('Распределение типов данных')

            # Качество по активностям (топ-8)
            plt.subplot(2, 2, 4)
            top_activities = self.df['activity'].value_counts().head(8).index
            activity_quality = []
            for activity in top_activities:
                activity_data = self.df[self.df['activity'] == activity]
                quality = 1 - activity_data.isnull().mean().mean()
                activity_quality.append(quality)

            plt.bar(range(len(top_activities)), activity_quality)
            plt.title('Качество данных по активностям (топ-8)')
            plt.xticks(range(len(top_activities)), top_activities, rotation=45)
            plt.ylabel('Показатель качества')

            plt.tight_layout()
            plt.savefig(f'{output_dir}/basic_characteristics.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Анализ выбросов и распределений
            plt.figure(figsize=(15, 10))

            # Распределение ускорений
            numerical_cols = ['accel_x', 'accel_y', 'accel_z']
            for i, col in enumerate(numerical_cols, 1):
                if col in self.df.columns:
                    plt.subplot(2, 3, i)
                    plt.hist(self.df[col].dropna(), bins=50, alpha=0.7)
                    plt.title(f'Распределение {col}')
                    plt.xlabel('Значение')
                    plt.ylabel('Частота')

            # Выбросы по осям
            plt.subplot(2, 3, 4)
            outlier_counts = []
            valid_columns = []
            for col in numerical_cols:
                if col in self.analysis_results['quality']['outliers']:
                    outlier_info = self.analysis_results['quality']['outliers'][col]
                    if 'outlier_count' in outlier_info:
                        outlier_counts.append(outlier_info['outlier_count'])
                        valid_columns.append(col)

            if valid_columns:
                plt.bar(valid_columns, outlier_counts)
                plt.title('Количество выбросов по осям акселерометра')
                plt.ylabel('Количество выбросов')

            # Матрица корреляции
            plt.subplot(2, 3, 5)
            if all(col in self.df.columns for col in numerical_cols):
                correlation_matrix = self.df[numerical_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Матрица корреляции акселерометра')

            # Величина ускорения
            plt.subplot(2, 3, 6)
            if all(col in self.df.columns for col in numerical_cols):
                acceleration_magnitude = np.sqrt(
                    self.df['accel_x'] ** 2 + self.df['accel_y'] ** 2 + self.df['accel_z'] ** 2)
                plt.hist(acceleration_magnitude, bins=50, alpha=0.7)
                plt.title('Распределение величины ускорения')
                plt.xlabel('Величина ускорения')
                plt.ylabel('Частота')

            plt.tight_layout()
            plt.savefig(f'{output_dir}/distributions_and_outliers.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Ошибка при создании визуализаций: {e}")

    def save_detailed_analysis(self, output_file='dataset_characteristics_detailed.json'):
        """Сохранение детального анализа в файл"""

        # Функция для сериализации numpy типов
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=convert_to_serializable)

    def run_complete_analysis(self):
        """Запуск полного анализа"""
        print("Запуск комплексного анализа характеристик датасета...")

        self.load_dataset()
        self.analyze_size_and_dimensions()
        self.analyze_data_types()
        self.analyze_data_sparsity()
        self.analyze_data_resolution()
        self.analyze_data_quality()
        self.analyze_usage_purpose()

        report = self.generate_comprehensive_report()
        self.create_detailed_visualizations()
        self.save_detailed_analysis()

        print("\nАнализ завершен!")
        print(f"Результаты сохранены в папке: dataset_characteristics/")
        print(f"Детальный отчет сохранен в: dataset_characteristics_detailed.json")

        return report


# Основной блок выполнения
def main():
    """Основная функция для запуска анализа"""
    dataset_path = "../HMP_Dataset"  # Укажите путь к датасету

    analyzer = DatasetCharacteristicsAnalyzer(dataset_path)

    try:
        report = analyzer.run_complete_analysis()
        print(report)

    except Exception as e:
        print(f"Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()