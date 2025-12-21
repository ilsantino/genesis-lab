# Dominio 3: Financial Transactions (Arquitectura Documentada)

## Estado
Este dominio está documentado arquitecturalmente pero no implementado en el MVP. El propósito de este documento es definir la arquitectura completa para que pueda ser implementado en versiones futuras del proyecto.

---
## Justificación de No Implementación
Durante la planificación del MVP tomamos la decisión estratégica de implementar dos dominios completamente en lugar de tres superficialmente. Customer service y time series proporcionan mejor diversidad técnica porque uno es texto conversacional y otro es datos numéricos temporales. Financial transactions sería otro dominio de datos tabulares similar en naturaleza técnica a time series.
Al enfocarnos en dos dominios podemos implementar features avanzadas como validación robusta, bias detection sofisticado, y análisis temporal profundo. Esto resulta en un proyecto de mayor calidad técnica que demuestra profundidad sobre amplitud, lo cual es más valioso tanto para portfolio académico como para uso real en iaGO.

---
## Schema de Datos
El schema para transacciones financieras sintéticas está definido en src slash generation slash schemas punto py con la clase FinancialTransaction. Cada transacción incluye transaction_id como identificador único, account_id para vincular con una cuenta, transaction_type que puede ser purchase, withdrawal, deposit, transfer o payment, amount como valor numérico positivo, timestamp para cuándo ocurrió la transacción, merchant_category opcional como grocery, gas o retail, location opcional con city, country y coordinates, is_fraudulent como boolean indicando si es fraude, risk_score entre cero y uno, y metadata como diccionario para información adicional.
La validación de Pydantic asegura que amount sea mayor que cero, que risk_score esté entre cero y uno, y que todos los campos requeridos estén presentes.

---
## Estrategia de Generación
Para generar transacciones financieras sintéticas realistas usaríamos prompts que enfatizan patrones financieros auténticos. El system prompt instruiría al LLM a generar transacciones que siguen comportamientos típicos de consumidores reales, incluyendo patrones temporales donde transacciones de grocery ocurren más frecuentemente en fines de semana, transacciones grandes son menos frecuentes que transacciones pequeñas, y fraud tiene patrones característicos como montos inusuales o locaciones geográficamente inconsistentes.
Los few-shot examples mostrarían tanto transacciones normales como fraudulentas. Para transacciones normales incluiríamos ejemplos de compra en grocery store con monto razonable, transferencia entre cuentas propias, pago de servicio recurrente, y retiro de cajero en locación familiar. Para transacciones fraudulentas incluiríamos ejemplos de compra con monto inusualmente alto, múltiples transacciones en locaciones geográficamente distantes en corto tiempo, transacción en merchant category inconsistente con historial del usuario, y transacción en horario atípico para ese tipo de comercio.
Los parámetros de generación incluirían fraud_rate de dos por ciento para aproximar tasas reales de fraude, daily_transaction_count entre uno y diez transacciones por cuenta por día, y amount_distribution con setenta por ciento de transacciones pequeñas entre cinco y cincuenta, veinticinco por ciento de transacciones medianas entre cincuenta y quinientos, y cinco por ciento de transacciones grandes entre quinientos y cinco mil.

---
## Métricas de Validación
Las métricas de calidad para transacciones financieras evaluarían realismo de distribución de amounts comparando con distribuciones de datasets financieros reales, consistencia temporal verificando que no haya patrones ilógicos como múltiples transacciones simultáneas o compras de grocery a las tres de la mañana, coherencia de merchant category asegurando que las categorías sean consistentes con los amounts y patterns de cada cuenta, y autenticidad de patrones de comportamiento comparando contra comportamiento conocido de consumidores reales.
Para fraud detection evaluaríamos precision y recall en las etiquetas de fraude, false positive rate porque en finanzas los falsos positivos son costosos, y AUC-ROC score como métrica estándar de clasificación binaria. El objetivo sería lograr precision mayor a noventa por ciento y recall mayor a ochenta por ciento.

---
## Tarea de Training
La tarea de machine learning sería clasificación binaria de fraud detection. Las features incluirían amount normalizado, time of day extraído del timestamp, day of week, merchant category codificado como one-hot encoding, transaction frequency calculado como rolling statistics de transacciones recientes, amount deviation calculado como qué tan desviado está el monto del promedio histórico del usuario, geographic consistency verificando si la locación es consistente con el historial, y temporal consistency verificando si el timing es consistente con patrones previos.
Los modelos a probar serían logistic regression como baseline simple, XGBoost como modelo primario porque maneja bien features categóricas y numéricas mezcladas, e isolation forest para anomaly detection como enfoque alternativo. Las métricas enfatizarían recall porque es más importante detectar fraud que minimizar false positives, junto con precision, F1 score, ROC-AUC, y confusion matrix.

---
## Bias Detection
Los sesgos potenciales a verificar incluirían bias geográfico verificando que no haya sobre-representación de ciertas regiones, bias de amounts verificando que la distribución siga una distribución realista y no esté artificialmente centrada, bias temporal verificando que las transacciones estén distribuidas realísticamente a lo largo del día y la semana, y bias de merchant category verificando que haya variedad suficiente en tipos de comercios.
Las consideraciones de fairness asegurarían que fraud no esté correlacionado con demographics si se incluyen en metadata, que haya representación balanceada de diferentes merchant categories, y que los patrones sean realistas para diferentes tipos de cuentas si se modelan cuentas personal versus business.

---
## Reference Dataset
El reference dataset sugerido sería card_fraud o creditcard_fraud de HuggingFace, o alternativamente el Credit Card Fraud Detection dataset de Kaggle que tiene más de doscientas mil transacciones reales con etiquetas de fraude. Este dataset se usaría para comparar distribuciones estadísticas de nuestras transacciones sintéticas versus transacciones reales, calibrar el fraud_rate para que coincida con tasas reales, y validar que nuestros patrones de fraude sean realistas comparados con fraude real.

---
## Roadmap de Implementación
La fase uno de implementación básica tomaría dos a tres días e incluiría implementar prompt templates, generar transacciones básicas, y validar distribuciones de amounts. La fase dos de patrones de fraude tomaría dos días e incluiría agregar patrones realistas de fraude, validar métricas de fraud detection, y tune fraud_rate.
La fase tres de features avanzadas tomaría dos días e incluiría agregar datos geográficos, implementar patrones temporales sofisticados, y agregar merchant categories detalladas. La fase cuatro de training y evaluación tomaría dos días e incluiría implementar modelos de fraud detection, comparar contra reference dataset, y optimizar para precision y recall.

---
## Desafíos Esperados
Los desafíos técnicos anticipados incluyen que LLMs pueden generar patrones de fraude demasiado obvios que serían fáciles de detectar, asegurar consistencia temporal de que las transacciones sigan patrones lógicos de tiempo, lograr distribuciones realistas de amounts porque distribuciones financieras tienen long tails complejas, y variedad limitada de merchant categories sin prompting extensivo.

---
## Criterios de Éxito
Los criterios de éxito si este dominio se implementara serían generar diez mil transacciones en menos de una hora, lograr AUC de fraud detection mayor a cero punto noventa, alcanzar quality score mayor a ochenta y cinco, no detectar biases demográficos, y lograr que la distribución de amounts esté dentro de diez por ciento de la distribución del reference dataset.

---
## Valor de Esta Documentación
Documentar este dominio aunque no lo implementemos demuestra pensamiento arquitectural completo, muestra que consideramos el problema holísticamente aunque priorizamos pragmáticamente, proporciona una base para implementación futura si el proyecto se extiende, y exhibe capacidad de diseño de sistemas sin necesidad de implementación inmediata, lo cual es una habilidad valiosa en roles de arquitectura.