#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <exception>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// ========================================================
// ФУНКЦИЯ РАСПРЕДЕЛЕНИЯ СТЬЮДЕНТА (CDF)
// ========================================================

double studentTCDF(double t, int df) {
    double x = df / (df + t * t);
    auto betacf = [](double a, double b, double x) {
        const int MAXIT = 100;
        const double EPS = 3e-7;
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (fabs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= MAXIT; ++m) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (fabs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (fabs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (fabs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (fabs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (fabs(del - 1.0) < EPS) break;
        }
        return h;
        };

    double a = df / 2.0;
    double b = 0.5;
    double bt = exp(
        lgamma(a + b) - lgamma(a) - lgamma(b)
        + a * log(x) + b * log(1.0 - x)
    );

    if (t >= 0)
        return 1.0 - 0.5 * bt * betacf(a, b, x);
    else
        return 0.5 * bt * betacf(a, b, x);
}

// ========================================================
// КЛАСС МНОГОФАКТОРНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ (ИЗ .DOCX)
// ========================================================

class MultipleLinearRegression {
private:
    MatrixXd X_original;
    MatrixXd X_with_const;
    VectorXd y_original;
    VectorXd beta;
    int n; // наблюдения
    int k; // коэффициенты (вкл. константу)
    vector<string> factor_names;

public:
    MultipleLinearRegression() : n(0), k(0) {}

    bool fit(const MatrixXd& X, const VectorXd& y,
        const vector<string>& names) {
        if (X.rows() != y.size()) return false;
        n = X.rows();
        int m = X.cols();
        k = m + 1;
        X_original = X;
        y_original = y;
        factor_names = names;

        X_with_const.resize(n, k);
        X_with_const.col(0) = VectorXd::Ones(n);
        X_with_const.block(0, 1, n, m) = X;

        beta = (X_with_const.transpose() * X_with_const)
            .inverse()
            * X_with_const.transpose() * y;

        return true;
    }

    VectorXd predict(const MatrixXd& X) const {
        MatrixXd Xp(X.rows(), k);
        Xp.col(0) = VectorXd::Ones(X.rows());
        Xp.block(0, 1, X.rows(), X.cols()) = X;
        return Xp * beta;
    }

    VectorXd getCoefficients() const {
        return beta;
    }

    vector<string> getFactorNames() const {
        return factor_names;
    }

    // ====================================================
    // РАСЧЁТ СТАТИСТИК
    // ====================================================
    void calculateStatistics(double& r2,
        double& adj_r2,
        double& rmse,
        double& mape,
        double& mae,
        VectorXd& std_errors,
        VectorXd& t_stats,
        VectorXd& p_values) {
        VectorXd y_pred = predict(X_original);
        VectorXd residuals = y_original - y_pred;

        double sse = residuals.squaredNorm();
        double sst = (y_original.array() - y_original.mean()).square().sum();
        r2 = 1.0 - sse / sst;
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k);
        rmse = sqrt(sse / n);
        mae = residuals.array().abs().mean();
        mape = 0.0;
        int cnt = 0;

        for (int i = 0; i < n; i++) {
            if (fabs(y_original[i]) > 1e-12) {
                mape += fabs(residuals[i] / y_original[i]);
                cnt++;
            }
        }
        if (cnt > 0) mape = mape / cnt * 100.0;

        MatrixXd XtX_inv = (X_with_const.transpose() * X_with_const).inverse();
        double sigma2 = sse / (n - k);
        std_errors = (sigma2 * XtX_inv.diagonal()).array().sqrt();

        // ===== КОРРЕКТНЫЕ t-СТАТИСТИКИ И p-value =====
        t_stats.resize(k);
        p_values.resize(k);
        int df = n - k;

        for (int i = 0; i < k; i++) {
            t_stats[i] = beta[i] / std_errors[i];
            double t_abs = fabs(t_stats[i]);
            double cdf = studentTCDF(t_abs, df);
            p_values[i] = 2.0 * (1.0 - cdf);
        }
    }

    // ====================================================
    // F-СТАТИСТИКА (классическая)
    // ====================================================
    double calculateFStatistic() {
        VectorXd y_pred = predict(X_original);
        VectorXd residuals = y_original - y_pred;
        double sse = residuals.squaredNorm();
        double ssr = (y_pred.array() - y_original.mean()).square().sum();
        return (ssr / (k - 1)) / (sse / (n - k));
    }

    // ====================================================
    // МАТРИЦА КОРРЕЛЯЦИЙ МЕЖДУ ФАКТОРАМИ
    // ====================================================
    MatrixXd calculateCorrelationMatrix() {
        int m = X_original.cols();
        MatrixXd corr = MatrixXd::Zero(m, m);

        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                VectorXd xi = X_original.col(i);
                VectorXd xj = X_original.col(j);
                double mi = xi.mean();
                double mj = xj.mean();
                double num = ((xi.array() - mi) * (xj.array() - mj)).sum();
                double di = sqrt((xi.array() - mi).square().sum());
                double dj = sqrt((xj.array() - mj).square().sum());

                if (di > 0 && dj > 0) {
                    corr(i, j) = num / (di * dj);
                    corr(j, i) = corr(i, j);
                }
            }
        }
        return corr;
    }

    // ====================================================
    // КОРРЕЛЯЦИЯ ФАКТОРОВ С ОТКЛИКОМ
    // ====================================================
    VectorXd calculateCorrelationWithResponse() {
        int m = X_original.cols();
        VectorXd corr(m);
        double my = y_original.mean();
        double dy = sqrt((y_original.array() - my).square().sum());

        for (int i = 0; i < m; i++) {
            VectorXd xi = X_original.col(i);
            double mi = xi.mean();
            double di = sqrt((xi.array() - mi).square().sum());

            if (di > 0 && dy > 0) {
                double num = ((xi.array() - mi) * (y_original.array() - my)).sum();
                corr[i] = num / (di * dy);
            }
            else {
                corr[i] = 0.0;
            }
        }
        return corr;
    }

    // ====================================================
    // ОТБОР ЗНАЧИМЫХ ФАКТОРОВ ПО p-value
    // ====================================================
    vector<int> selectSignificantFactors(const VectorXd& p_values,
        double alpha) {
        vector<int> result;
        for (int i = 1; i < p_values.size(); i++) {
            if (p_values[i] < alpha) {
                result.push_back(i - 1);
            }
        }
        return result;
    }

    // ====================================================
    // ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ
    // ====================================================
    vector<int> checkMulticollinearity(double threshold) {
        MatrixXd corr = calculateCorrelationMatrix();
        VectorXd corr_y = calculateCorrelationWithResponse();
        vector<int> to_remove;

        for (int i = 0; i < corr.rows(); i++) {
            for (int j = i + 1; j < corr.cols(); j++) {
                if (fabs(corr(i, j)) > threshold) {
                    if (fabs(corr_y[i]) < fabs(corr_y[j])) {
                        if (find(to_remove.begin(), to_remove.end(), i) == to_remove.end())
                            to_remove.push_back(i);
                    }
                    else {
                        if (find(to_remove.begin(), to_remove.end(), j) == to_remove.end())
                            to_remove.push_back(j);
                    }
                }
            }
        }
        return to_remove;
    }

    // ====================================================
    // УДАЛЕНИЕ ФАКТОРОВ И СОЗДАНИЕ НОВОЙ МОДЕЛИ
    // ====================================================
    MultipleLinearRegression removeFactors(const vector<int>& idx) {
        if (idx.empty()) return *this;

        int new_m = X_original.cols() - idx.size();
        MatrixXd X_new(n, new_m);
        vector<string> names_new;
        int c = 0;

        for (int i = 0; i < X_original.cols(); i++) {
            if (find(idx.begin(), idx.end(), i) == idx.end()) {
                X_new.col(c) = X_original.col(i);
                names_new.push_back(factor_names[i]);
                c++;
            }
        }

        MultipleLinearRegression model;
        model.fit(X_new, y_original, names_new);
        return model;
    }
};

// ============================================================
// КЛАСС ДЛЯ РАБОТЫ С ДАННЫМИ РОССТАТА (ИЗ .CPP)
// ============================================================

class RosstatData {
private:
    struct TimeSeries {
        string region_name;
        string region_code;
        vector<double> values;
    };

    vector<TimeSeries> series_data;
    vector<string> time_periods;
    vector<string> factor_names;
    vector<int> selected_years;
    int target_year_idx;

public:
    RosstatData() : target_year_idx(-1) {}

    // Чтение данных из CSV файла Росстата
    bool loadFromCSV(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Ошибка: не удалось открыть файл " << filename << endl;
            return false;
        }

        string line;
        int line_count = 0;

        while (getline(file, line)) {
            line_count++;

            if (line.empty() || line.find_first_not_of(';') == string::npos) {
                continue;
            }

            while (!line.empty() && line.back() == ';') {
                line.pop_back();
            }

            vector<string> tokens;
            stringstream ss(line);
            string token;

            while (getline(ss, token, ';')) {
                tokens.push_back(token);
            }

            if (tokens.size() < 3) continue;

            if (line_count == 1) {
                continue;
            }
            else if (line_count == 2) {
                for (size_t i = 2; i < tokens.size(); i++) {
                    string year_str = tokens[i];
                    size_t pos = year_str.find(" г.");
                    if (pos != string::npos) {
                        year_str = year_str.substr(0, pos);
                    }
                    time_periods.push_back(year_str);
                }
            }
            else {
                TimeSeries ts;
                ts.region_name = tokens[0];
                ts.region_code = tokens[1];

                for (size_t i = 2; i < tokens.size(); i++) {
                    string val_str = tokens[i];

                    val_str.erase(remove(val_str.begin(), val_str.end(), ' '), val_str.end());
                    val_str.erase(remove(val_str.begin(), val_str.end(), '\"'), val_str.end());

                    size_t comma_pos = val_str.find(',');
                    if (comma_pos != string::npos) {
                        val_str[comma_pos] = '.';
                    }

                    string cleaned;
                    for (char c : val_str) {
                        if (c != ' ') cleaned += c;
                    }

                    try {
                        if (!cleaned.empty() && cleaned != "-" && cleaned != "…" &&
                            cleaned != ".." && cleaned != "\"\"" && cleaned != ".") {
                            ts.values.push_back(stod(cleaned));
                        }
                        else {
                            ts.values.push_back(numeric_limits<double>::quiet_NaN());
                        }
                    }
                    catch (...) {
                        ts.values.push_back(numeric_limits<double>::quiet_NaN());
                    }
                }

                int valid_count = 0;
                for (double val : ts.values) {
                    if (!isnan(val)) valid_count++;
                }

                if (valid_count >= 5) {
                    series_data.push_back(ts);
                }
            }
        }

        file.close();

        if (series_data.empty()) {
            cerr << "Ошибка: не удалось загрузить данные из файла." << endl;
            return false;
        }

        cout << "Загружено временных рядов: " << series_data.size() << endl;
        cout << "Периодов данных: " << time_periods.size() << endl;

        if (!series_data.empty()) {
            cout << "Пример региона: " << series_data[0].region_name << endl;
        }

        return true;
    }

    // Подготовка данных для регрессионного анализа
    bool prepareRegressionData(MatrixXd& X, VectorXd& y, int num_factors = 5) {
        vector<vector<double>> X_vec;
        vector<double> y_vec;

        target_year_idx = time_periods.size() - 1;

        factor_names.clear();
        selected_years.clear();

        for (int i = 0; i < num_factors; i++) {
            selected_years.push_back(target_year_idx - i - 1);
            factor_names.push_back("Year_" + time_periods[target_year_idx - i - 1]);
        }

        for (const auto& ts : series_data) {
            bool valid = true;
            vector<double> x_row;

            for (int year_idx : selected_years) {
                if (year_idx >= 0 && year_idx < (int)ts.values.size() &&
                    !isnan(ts.values[year_idx])) {
                    x_row.push_back(ts.values[year_idx]);
                }
                else {
                    valid = false;
                    break;
                }
            }

            if (valid && target_year_idx < (int)ts.values.size() &&
                !isnan(ts.values[target_year_idx])) {
                X_vec.push_back(x_row);
                y_vec.push_back(ts.values[target_year_idx]);
            }
        }

        if (X_vec.size() < 5) {
            cerr << "Ошибка: недостаточно данных для анализа ("
                << X_vec.size() << " наблюдений)" << endl;
            return false;
        }

        int n = X_vec.size();
        int m = num_factors;

        X.resize(n, m);
        y.resize(n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                X(i, j) = X_vec[i][j];
            }
            y(i) = y_vec[i];
        }

        cout << "\nПодготовка данных для регрессии:" << endl;
        cout << "  Отклик (Y): данные за " << time_periods[target_year_idx] << " год" << endl;
        cout << "  Факторы (X): " << m << " предыдущих лет" << endl;
        cout << "  Наблюдений: " << n << endl;

        return true;
    }

    vector<string> getFactorNames() const {
        return factor_names;
    }

    int getTargetYearIdx() const {
        return target_year_idx;
    }

    string getTargetYear() const {
        if (target_year_idx >= 0 && target_year_idx < (int)time_periods.size()) {
            return time_periods[target_year_idx];
        }
        return "";
    }

    vector<string> getRegionNames() const {
        vector<string> names;
        for (const auto& ts : series_data) {
            names.push_back(ts.region_name);
        }
        return names;
    }

    const TimeSeries& getRegionData(int idx) const {
        return series_data[idx];
    }
};

// ============================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВЫВОДА (ИЗ .DOCX)
// ============================================================

void printCorrelationMatrix(const MatrixXd& corr,
    const vector<string>& names) {
    cout << "\nМатрица корреляций между факторами:\n";
    cout << setw(15) << " ";
    for (const auto& n : names)
        cout << setw(12) << n.substr(0, 10);
    cout << "\n";

    for (int i = 0; i < corr.rows(); i++) {
        cout << setw(15) << names[i].substr(0, 10);
        for (int j = 0; j < corr.cols(); j++) {
            cout << setw(12) << fixed << setprecision(4) << corr(i, j);
        }
        cout << "\n";
    }
}

void printCorrelationWithResponse(const VectorXd& corr,
    const vector<string>& names) {
    cout << "\nКорреляции факторов с откликом:\n";
    for (int i = 0; i < corr.size(); i++) {
        cout << setw(20) << left << names[i]
            << ": " << fixed << setprecision(4) << corr[i] << "\n";
    }
}

// ============================================================
// СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В ФАЙЛ (ИЗ .DOCX)
// ============================================================

void saveResultsToFile(const string& filename,
    const VectorXd& coef,
    const vector<string>& names,
    double r2, double adj_r2,
    double rmse, double mape, double mae,
    double f_stat,
    const VectorXd& p_values) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка сохранения файла\n";
        return;
    }

    file << fixed << setprecision(6);
    file << "РЕЗУЛЬТАТЫ РЕГРЕССИОННОГО АНАЛИЗА\n";
    file << "=================================\n\n";

    file << "Коэффициенты модели:\n";
    file << "Константа: " << coef[0]
        << " (p = " << p_values[0] << ")\n";

    for (size_t i = 0; i < names.size(); i++) {
        file << names[i] << ": "
            << coef[i + 1]
            << " (p = " << p_values[i + 1] << ")\n";
    }

    file << "\nКачество модели:\n";
    file << "R2: " << r2 << "\n";
    file << "R2 adj: " << adj_r2 << "\n";
    file << "F-stat: " << f_stat << "\n";
    file << "RMSE: " << rmse << "\n";
    file << "MAE: " << mae << "\n";
    file << "MAPE: " << mape << "%\n";

    file.close();
}

// ============================================================
// ГЛАВНАЯ ФУНКЦИЯ (КОМБИНИРОВАННАЯ)
// ============================================================

int main() {
    setlocale(LC_ALL, "Russian");

    cout << "=============================================\n";
    cout << "МНОГОФАКТОРНАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ\n";
    cout << "=============================================\n\n";

    try {
        // ВЫБОР ФАЙЛА С ДАННЫМИ
        cout << "Введите имя файла с данными (по умолчанию DataV5.csv): ";
        string filename;
        getline(cin, filename);

        if (filename.empty()) {
            filename = "DataV5.csv";
        }

        // ЗАГРУЗКА ДАННЫХ
        cout << "\n1. ЗАГРУЗКА ДАННЫХ\n";
        cout << "------------------\n";

        RosstatData data;
        if (!data.loadFromCSV(filename)) {
            cerr << "Ошибка загрузки данных.\n";
            return 1;
        }

        // ПОДГОТОВКА ДАННЫХ
        cout << "\n2. ПОДГОТОВКА ДАННЫХ\n";
        cout << "--------------------\n";

        MatrixXd X;
        VectorXd y;

        int num_factors = 5;
        cout << "Введите количество факторов (по умолчанию 5): ";
        string input;
        getline(cin, input);

        if (!input.empty()) {
            try {
                num_factors = stoi(input);
                if (num_factors < 2) num_factors = 2;
                if (num_factors > 10) num_factors = 10;
            }
            catch (...) {
                num_factors = 5;
            }
        }

        if (!data.prepareRegressionData(X, y, num_factors)) {
            cerr << "Ошибка подготовки данных для регрессии.\n";
            return 1;
        }

        // НАСТРОЙКА ПАРАМЕТРОВ
        cout << "\n3. НАСТРОЙКА ПАРАМЕТРОВ\n";
        cout << "----------------------\n";

        double significance_level = 0.05;
        cout << "Введите уровень значимости (по умолчанию 0.05): ";
        getline(cin, input);

        if (!input.empty()) {
            try {
                significance_level = stod(input);
                if (significance_level <= 0) significance_level = 0.05;
                if (significance_level >= 1) significance_level = 0.05;
            }
            catch (...) {
                significance_level = 0.05;
            }
        }

        double correlation_threshold = 0.8;
        cout << "Введите порог мультиколлинеарности (по умолчанию 0.8): ";
        getline(cin, input);

        if (!input.empty()) {
            try {
                correlation_threshold = stod(input);
                if (correlation_threshold <= 0) correlation_threshold = 0.8;
                if (correlation_threshold >= 1) correlation_threshold = 0.8;
            }
            catch (...) {
                correlation_threshold = 0.8;
            }
        }

        // ОБУЧЕНИЕ МОДЕЛИ
        cout << "\n4. ОБУЧЕНИЕ МОДЕЛИ\n";
        cout << "------------------\n";

        vector<string> factor_names = data.getFactorNames();
        MultipleLinearRegression model;

        if (!model.fit(X, y, factor_names)) {
            cerr << "Не удалось обучить модель.\n";
            return 1;
        }

        cout << "Модель успешно обучена\n";

        // РАСЧЁТ СТАТИСТИК
        double r2, adj_r2, rmse, mape, mae;
        VectorXd se, t, p;
        model.calculateStatistics(r2, adj_r2, rmse, mape, mae, se, t, p);
        double f_stat = model.calculateFStatistic();

        // ВЫВОД РЕЗУЛЬТАТОВ
        cout << "\n5. РЕЗУЛЬТАТЫ АНАЛИЗА\n";
        cout << "--------------------\n";

        cout << "\nКоэффициенты:\n";
        VectorXd coef = model.getCoefficients();
        cout << "Const: " << coef[0]
            << " (p=" << p[0] << ")\n";

        for (size_t i = 0; i < factor_names.size(); i++) {
            cout << factor_names[i] << ": "
                << coef[i + 1]
                << " (p=" << p[i + 1] << ")\n";
        }

        cout << "\nR2 = " << r2
            << "\nR2 adj = " << adj_r2
            << "\nF = " << f_stat
            << "\nRMSE = " << rmse
            << "\nMAE = " << mae
            << "\nMAPE = " << mape << "%\n";

        // ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ
        cout << "\n6. ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ\n";
        cout << "-------------------------------\n";

        MatrixXd corr_matrix = model.calculateCorrelationMatrix();
        printCorrelationMatrix(corr_matrix, factor_names);

        VectorXd corr_y = model.calculateCorrelationWithResponse();
        printCorrelationWithResponse(corr_y, factor_names);

        vector<int> multicollinear = model.checkMulticollinearity(correlation_threshold);
        if (!multicollinear.empty()) {
            cout << "\nОбнаружена мультиколлинеарность у факторов:\n";
            for (int idx : multicollinear) {
                cout << "  - " << factor_names[idx] << endl;
            }

            cout << "\nХотите удалить проблемные факторы? (y/n): ";
            getline(cin, input);

            if (!input.empty() && tolower(input[0]) == 'y') {
                MultipleLinearRegression new_model = model.removeFactors(multicollinear);
                model = new_model;
                factor_names = model.getFactorNames();

                // Пересчитываем статистики
                model.calculateStatistics(r2, adj_r2, rmse, mape, mae, se, t, p);
                f_stat = model.calculateFStatistic();
                coef = model.getCoefficients();

                cout << "\nМодель обновлена. Новые коэффициенты:\n";
                cout << "Const: " << coef[0] << " (p=" << p[0] << ")\n";
                for (size_t i = 0; i < factor_names.size(); i++) {
                    cout << factor_names[i] << ": " << coef[i + 1] << " (p=" << p[i + 1] << ")\n";
                }
            }
        }

        // ОТБОР ЗНАЧИМЫХ ФАКТОРОВ
        cout << "\n7. ОТБОР ЗНАЧИМЫХ ФАКТОРОВ\n";
        cout << "-------------------------\n";

        vector<int> significant = model.selectSignificantFactors(p, significance_level);
        if (!significant.empty()) {
            cout << "Значимые факторы (p < " << significance_level << "):\n";
            for (int idx : significant) {
                cout << "  ok " << factor_names[idx]
                    << " (p = " << scientific << setprecision(2) << p[idx + 1] << ")\n";
            }
        }
        else {
            cout << "Нет значимых факторов на уровне " << significance_level << "\n";
        }

        // СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        cout << "\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ\n";
        cout << "-----------------------\n";

        saveResultsToFile("regression_results_v5.txt",
            coef, factor_names,
            r2, adj_r2,
            rmse, mape, mae,
            f_stat, p);

        cout << "Анализ завершён корректно\n";
        cout << "Результаты сохранены в файл: regression_results_v5.txt\n";

    }
    catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
