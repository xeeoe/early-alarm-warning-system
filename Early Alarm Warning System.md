# Early Alarm Warning System

## Bilingual Project Description and User Guide

---

# العربية

## وصف المشروع

**Early Alarm Warning System** هو تطبيق ويب تفاعلي مبني باستخدام **Streamlit**، ويهدف إلى تحليل المؤشرات الاقتصادية واستخدام نموذج تعلم آلي مبني على **XGBoost** للتنبؤ باحتمالية حدوث أزمة اقتصادية خلال الاثني عشر شهرًا التالية.

يجمع التطبيق بين تحليل بيانات التدريب، وتجهيز بيانات جديدة بطريقة مرنة، وتنفيذ التوقعات اليدوية أو التوقعات على ملفات CSV، وعرض نتائج توقعات جاهزة. كما يوفر مجموعة من الرسومات التفاعلية التي تساعد على فهم البيانات والعلاقات بين المؤشرات الاقتصادية.

> **تنبيه مهم:** هذا التطبيق أداة إنذار مبكر ومساعدة في التحليل، وليس نظامًا دقيقًا أو بديلًا عن التحقق البشري والتحليل الاقتصادي المتخصص. إذا ظهرت نسبة خطر مرتفعة، فيجب على المحلل مراجعة البيانات والتأكد من جودتها ومقارنتها بمصادر ومؤشرات إضافية، ثم اتخاذ القرار المناسب بناءً على خبرته وتحليله. لا ينبغي اتخاذ قرار اقتصادي أو مالي اعتمادًا على نتيجة النموذج وحدها.

## الإفصاح عن استخدام الذكاء الاصطناعي

تم تطوير ملف `crisis_prediction_pipeline_xgb_fixed_no_nulls.ipynb` بمساعدة أدوات الذكاء الاصطناعي، وذلك للمساعدة في كتابة أجزاء من الكود وشرح خطوات تجهيز البيانات والتنبؤ، نظرًا لمحدودية الخبرة البرمجية لدى صاحب المشروع في بعض الجوانب. تمت مراجعة الكود ودمجه ضمن المشروع بهدف التعلم والتطوير، لكن يجب اختبار النتائج والتحقق منها قبل الاعتماد عليها. تظل المسؤولية النهائية عن مراجعة الكود والبيانات والنتائج واستخدام التطبيق على عاتق صاحب المشروع والمستخدم النهائي.

## أهم خصائص التطبيق

| القسم | الوظيفة |
|---|---|
| **Training Data Analysis** | تحليل بيانات التدريب وعرض ملخص البيانات والقيم المفقودة والتكرارات والرسومات الإحصائية. |
| **New Data Prediction** | رفع أي ملف CSV جديد أو إدخال حالة اقتصادية يدويًا وتشغيل النموذج عليها. |
| **Ready Prediction Viewer** | عرض ملف توقعات جاهز دون إعادة تشغيل النموذج عليه. |
| **Model Performance** | عرض Accuracy وPrecision وRecall وF1 Score ومصفوفة الالتباس وفجوة الأداء بين التدريب والاختبار. |

يعرض قسم التحليل **Heatmap قبل Feature Engineering وبعدها**، كما يكتشف أزواج الأعمدة ذات الارتباط العالي عندما تكون قيمة الارتباط المطلقة أكبر من أو تساوي `0.90`.

وتشمل الرسومات التفاعلية Histogram وBoxplot وScatter Matrix وHeatmap. ويمكن للمستخدم تمرير المؤشر لرؤية القيم، والتكبير، والتحريك، واستخدام شريط أدوات Plotly.

## طريقة عمل التنبؤ

يتعرف التطبيق على Features التي تدرب عليها النموذج، ثم يعيد ترتيب الأعمدة بنفس ترتيب التدريب. عند توفر أعمدة الدولة والسنة، يمكنه ترتيب البيانات وحساب التغيرات السنوية لبعض المؤشرات، مثل:

```text
inflation_change
gdp_growth_change
```

بعد ذلك يحول الأعمدة المطلوبة إلى قيم رقمية، وينشئ الأعمدة الناقصة عند الحاجة، ثم يستخدم الـ **fitted imputer** الموجود داخل النموذج لمعالجة القيم المفقودة قبل تشغيل التنبؤ.

يضيف التطبيق إلى النتيجة ثلاثة أعمدة أساسية:

```text
crisis_probability
crisis_next_12m_prediction
top_influencing_feature
```

يمثل `crisis_probability` احتمال الأزمة كنسبة مئوية، بينما يمثل `crisis_next_12m_prediction` التصنيف النهائي. أما `top_influencing_feature` فهو العمود صاحب أكبر مساهمة محلية في قرار كل صف عند توفر مساهمات XGBoost.

## المتطلبات

يحتاج التطبيق إلى Python إصدار 3.10 أو 3.11، بالإضافة إلى المكتبات الموجودة في ملف `requirements.txt`، ومنها Streamlit وPandas وNumPy وScikit-learn وXGBoost وPlotly وJoblib.

## ملفات المشروع

ضع الملفات التالية في مجلد واحد عند التشغيل المحلي أو داخل مستودع GitHub عند النشر:

```text
early_alarm_warning_system.py
requirements.txt
best_model_xgb_optimization.joblib
global_crisis_new.csv
crisis_predictions_full.csv
economic_indicators_2000_2025_with_fedfunds.csv
```

الملفات الثلاثة الأخيرة يمكن استخدامها بحسب القسم المطلوب. ملف النموذج `best_model_xgb_optimization.joblib` أساسي لتشغيل التوقعات.

## التشغيل على Windows

افتح PowerShell أو Terminal داخل مجلد المشروع، ثم نفذ الأوامر التالية:

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run early_alarm_warning_system.py
```

بعد التشغيل افتح الرابط الذي يظهر في الطرفية، وغالبًا يكون:

```text
http://localhost:8501
```

لإيقاف التطبيق اضغط `Ctrl + C` داخل Terminal.

## التشغيل على Linux أو macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run early_alarm_warning_system.py
```

## استخدام التطبيق

### 1. Training Data Analysis

اختر قسم **Training Data Analysis** من القائمة الجانبية. يمكنك استخدام ملف التدريب الافتراضي أو رفع ملف CSV آخر للتحليل. كما يمكنك رفع ملف Notebook بصيغة `.ipynb` أو ملف Python بصيغة `.py` لعرض محتواه والتعرف على خطوات التحليل الموجودة فيه.

يعرض القسم ملخص البيانات، وأنواع الأعمدة، والقيم المفقودة، والتكرارات، وHeatmap قبل وبعد التجهيز، وأزواج الارتباط العالي، وتوزيع الهدف، وHistograms، وBoxplots، وتحليل الدول، وOutlier Summary، وScatter Matrix عند توفر الأعمدة المناسبة.

### 2. New Data Prediction

اختر قسم **New Data Prediction**، ثم اختر إحدى الطريقتين:

**CSV prediction:** ارفع ملف CSV جديدًا. لا يشترط أن يكون الملف مطابقًا حرفيًا لملف سابق، لكن يجب أن يحتوي قدر الإمكان على Features التي تدرب عليها النموذج. سيظهر تحذير إذا كانت بعض الأعمدة ناقصة.

**Manual single prediction:** أدخل قيم Features يدويًا، ثم أدخل اسم البلد وISO3 والسنة إذا رغبت. اضغط **Run manual prediction** لعرض النتيجة وتنزيلها بصيغة CSV.

### 3. Ready Prediction Viewer

اختر قسم **Ready Prediction Viewer** لعرض ملف `crisis_predictions_full.csv` الجاهز، أو ارفع ملف Prediction آخر. هذا القسم يعرض النتائج الموجودة فقط ولا يعيد تشغيل النموذج عليها.

يمكنك عرض Heatmap للأعمدة الرقمية أو Feature Importance العامة للنموذج، ثم تنزيل الملف المعروض.

### 4. Model Performance

يعرض هذا القسم مقاييس الأداء على بيانات التدريب والاختبار، ومنها Accuracy وPrecision وRecall وF1 Score ومصفوفة الالتباس وفجوة Train-Test.

بالنسبة إلى Class 1، يوضح **Precision** نسبة الحالات التي صنفها النموذج كأزمات وكانت أزمات فعلًا، بينما يوضح **Recall** نسبة الأزمات الحقيقية التي استطاع النموذج اكتشافها. في نظام الإنذار المبكر، يُعد Recall مهمًا لأنه يوضح عدد الأزمات التي لم يفوّتها النموذج.

## حقوق الطبع والخصوصية

هذا المشروع من إعداد وتطوير صاحب المستودع. جميع حقوق الكود، وتنظيم التطبيق، والتعديلات، والتوثيق، وطريقة دمج التحليل مع نموذج التنبؤ محفوظة لصاحب المشروع ما لم يُذكر خلاف ذلك في ملف ترخيص مستقل.

لا يُسمح بنسخ الكود أو إعادة نشره أو استخدامه تجاريًا أو نسبه إلى شخص آخر دون إذن كتابي من صاحب المشروع. لا تستخدم هذا المستودع كترخيص تلقائي لإعادة الاستخدام؛ فغياب ملف ترخيص مفتوح المصدر يعني أن الحقوق محفوظة افتراضيًا لصاحب العمل وفق القوانين المحلية المعمول بها.

لحماية نسبة المشروع عمليًا، يُنصح بالاحتفاظ بنسخ مؤرخة من الكود وملفات التحليل، وتسجيل التعديلات في Git، وإضافة اسم المؤلف وحقوق الطبع داخل الملفات، والاحتفاظ بإثباتات إنشاء المشروع مثل لقطات الشاشة أو سجل المستودع أو ملفات التسليم المؤرخة. يمكن أيضًا إضافة اسم الحساب أو رابط المشروع الرسمي داخل واجهة التطبيق وملفات README.

يُفضّل عدم رفع النموذج أو ملفات البيانات الحساسة أو أي مفاتيح سرية إلى مستودع عام. عند مشاركة نسخة عامة، احذف البيانات الخاصة والمفاتيح والملفات التي لا تريد كشفها، واستخدم مستودعًا خاصًا إذا كانت الملفات غير مخصصة للنشر العام.

> هذا القسم معلومات عامة وليس استشارة قانونية. للحصول على حماية رسمية أو صياغة ترخيص مناسبة لبلدك، راجع محامي ملكية فكرية أو الجهة الرسمية المختصة بحقوق المؤلف.

## ملاحظات مهمة

تأكد من أن وحدات القياس في البيانات الجديدة مطابقة للوحدات المستخدمة أثناء تدريب النموذج. كما يجب عدم تغيير أسماء Features المطلوبة إلا إذا كنت ستضيف منطق تحويل مناسب.

إذا ظهرت رسالة تفيد بأن الملف غير موجود، راجع أسماء الملفات ومكانها. وإذا ظهرت مشكلة في تحميل نموذج Joblib، استخدم إصدار Python وXGBoost متوافقًا مع البيئة التي حُفظ فيها النموذج.

---

# English

## Project Description

**Early Alarm Warning System** is an interactive web application built with **Streamlit**. It analyzes economic indicators and uses an **XGBoost** machine-learning model to estimate the probability of an economic crisis occurring within the next twelve months.

The application combines training-data analysis, flexible preprocessing for new datasets, manual and CSV-based prediction, ready-result viewing, model-performance reporting, and interactive visualizations for exploring economic indicators.

> **Important warning:** This application is an early-warning and analytical-support tool, not a fully accurate decision system and not a substitute for human verification or professional economic analysis. When the model reports a high risk probability, the analyst should review the underlying data, validate its quality, compare the result with additional sources and indicators, and then make a decision based on professional judgment. Economic, financial, or policy decisions should never rely on the model output alone.

## AI-Assisted Development Disclosure

The file `crisis_prediction_pipeline_xgb_fixed_no_nulls.ipynb` was developed with assistance from artificial-intelligence tools. The assistance was used to help write parts of the code and explain data-preprocessing and prediction steps, particularly because the project owner had limited programming experience in some areas. The code was reviewed and integrated for learning and development purposes, but its outputs should be tested and independently validated before being relied upon. Final responsibility for reviewing the code, data, results, and use of the application remains with the project owner and the end user.

## Main Features

| Section | Function |
|---|---|
| **Training Data Analysis** | Explores the training dataset, missing values, duplicates, distributions, outliers, and engineered features. |
| **New Data Prediction** | Accepts a new CSV file or a manually entered economic case and applies the trained model. |
| **Ready Prediction Viewer** | Displays an existing prediction-results CSV without retraining or re-predicting it. |
| **Model Performance** | Reports Accuracy, Precision, Recall, F1 Score, confusion matrices, and train-test performance gaps. |

The analysis section includes correlation heatmaps **before and after feature engineering**. It also detects highly correlated feature pairs where the absolute correlation is greater than or equal to `0.90`.

Interactive visualizations include Histograms, Boxplots, Scatter Matrices, and Heatmaps. Plotly controls allow users to inspect values, zoom, pan, and interact with chart traces.

## Prediction Workflow

The application identifies the Features expected by the saved model and restores them in the same order used during training. When country and year columns are available, the application can sort observations and calculate annual changes for selected indicators, including:

```text
inflation_change
gdp_growth_change
```

The required Features are converted to numeric values. Missing model columns are created when necessary, and the fitted imputer stored inside the saved model pipeline is used to handle missing values before prediction.

Prediction outputs include:

```text
crisis_probability
crisis_next_12m_prediction
top_influencing_feature
```

`crisis_probability` represents the estimated crisis probability as a percentage. `crisis_next_12m_prediction` contains the final binary classification. `top_influencing_feature` represents the feature with the largest local contribution for each row when XGBoost contributions are available.

## Requirements

The application requires Python 3.10 or 3.11 and the packages listed in `requirements.txt`, including Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Plotly, Joblib, and imbalanced-learn.

## Project Files

Place the following files in one folder for local execution or in one GitHub repository for deployment:

```text
early_alarm_warning_system.py
requirements.txt
best_model_xgb_optimization.joblib
global_crisis_new.csv
crisis_predictions_full.csv
economic_indicators_2000_2025_with_fedfunds.csv
```

The saved model file `best_model_xgb_optimization.joblib` is required for prediction. The CSV files are used by the analysis, prediction, and ready-viewer sections as applicable.

## Windows Installation and Execution

Open PowerShell or a terminal inside the project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run early_alarm_warning_system.py
```

Open the URL displayed in the terminal. It is usually:

```text
http://localhost:8501
```

To stop the application, press `Ctrl + C` in the terminal.

## Linux or macOS Installation and Execution

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run early_alarm_warning_system.py
```

## Using the Application

### 1. Training Data Analysis

Select **Training Data Analysis** from the sidebar. You may use the default training dataset or upload another CSV file for analysis. You may also upload a Jupyter Notebook file (`.ipynb`) or Python file (`.py`) to display its source code and detect analysis steps.

The section provides dataset summaries, column types, missing-value statistics, duplicate counts, before-and-after heatmaps, highly correlated pairs, target distribution, Histograms, Boxplots, country analysis, outlier summaries, and a Scatter Matrix when suitable columns are available.

### 2. New Data Prediction

Select **New Data Prediction** and choose one of the following modes:

**CSV prediction:** Upload a new CSV file. The file does not need to be an exact copy of a previous dataset, but it should contain as many model Features as possible. The application warns you when expected columns are missing.

**Manual single prediction:** Enter the model Features manually, then optionally provide a country name, ISO3 code, and year. Click **Run manual prediction** to display and download the result as a CSV file.

### 3. Ready Prediction Viewer

Select **Ready Prediction Viewer** to display `crisis_predictions_full.csv`, or upload another prediction-results CSV. This section displays existing results and does not run the model again.

You can view a Heatmap of numeric prediction columns or the model’s global Feature Importance, then download the displayed file.

### 4. Model Performance

The performance section reports training and testing metrics, including Accuracy, Precision, Recall, F1 Score, confusion matrices, and the train-test performance gap.

For Class 1, **Precision** indicates how many cases predicted as crises were actually crises. **Recall** indicates how many of the real crisis cases were detected by the model. Recall is especially important in an early-warning system because it reflects how many real crises were not missed.

## Copyright, Attribution, and Privacy

This project was created and developed by the repository owner. All rights to the source code, application structure, modifications, documentation, and the integration of the analysis and prediction workflow are reserved by the project owner unless a separate license explicitly states otherwise.

The code may not be copied, redistributed, commercially used, or presented as someone else’s work without written permission from the project owner. This repository should not be treated as an automatic open-source license. If no open-source license is provided, copyright is generally reserved by the creator, subject to the laws applicable in the relevant jurisdiction.

To strengthen practical proof of authorship, keep dated versions of the source code and notebooks, maintain a Git history, include the author name and copyright notice inside the files, and preserve evidence of creation such as dated screenshots, repository history, or dated delivery records. You may also place the author name or official project URL in the application interface and README files.

Do not publish private datasets, confidential model files, API keys, passwords, or other sensitive material in a public repository. Remove private content before sharing a public version, or use a private repository when the files are not intended for public distribution.

> This section provides general information and is not legal advice. For formal copyright registration or a jurisdiction-specific license, consult an intellectual-property lawyer or the relevant copyright authority.

## Important Notes

Make sure that the units and definitions of new economic indicators match those used during model training. Do not rename required Features unless an appropriate transformation is implemented.

If the application reports that a file cannot be found, verify the filenames and their locations. If Joblib fails to load the model, use a compatible Python and XGBoost environment matching the environment used to save the model.

## License and Intended Use

Add a project-specific license before public distribution if required. Unless a separate license is provided, this repository should be treated as an academic or demonstration project and should not be used as the sole basis for economic, financial, or policy decisions.

## References

[1]: https://streamlit.io/ "Streamlit official website"
[3]: https://xgboost.readthedocs.io/ "XGBoost documentation"
[4]: https://plotly.com/python/ "Plotly Python documentation"
