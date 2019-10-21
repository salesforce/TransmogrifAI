/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.insights

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.feature.{DateListPivot, TransmogrifierDefaults}
import com.salesforce.op.stages.impl.insights.RecordInsightsParser.Insights
import com.salesforce.op.stages.impl.preparators.{SanityCheckDataTest, SanityChecker}
import com.salesforce.op.stages.impl.regression.OpLinearRegression
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.sparkwrappers.specific.OpPredictorWrapperModel
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomIntegral, RandomMap, RandomReal, RandomText, RandomVector}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnHistory, OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.{FeatureHistory, OpWorkflow, _}
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Encoder, Row}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Suite}


@RunWith(classOf[JUnitRunner])
class RecordInsightsLOCOTest extends FlatSpec with TestSparkContext with RecordInsightsTestDataGenerator {

  // scalastyle:off
  val data = Seq( // name, age, height, height_null, isBlueEyed, gender, testFeatNegCor
    SanityCheckDataTest("a", 32, 5.0, 0, 0.9, 0.5, 0),
    SanityCheckDataTest("b", 32, 4.0, 1, 0.1, 0, 0.1),
    SanityCheckDataTest("a", 32, 6.0, 0, 0.8, 0.5, 0),
    SanityCheckDataTest("a", 32, 5.5, 0, 0.85, 0.5, 0),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.2, 0, 0.1),
    SanityCheckDataTest("a", 32, 5.0, 0, 0.99, 0.5, 0),
    SanityCheckDataTest("b", 32, 4.0, 0, 0.0, 0, 0.1),
    SanityCheckDataTest("a", 32, 6.0, 1, 0.7, 0.5, 0),
    SanityCheckDataTest("a", 32, 5.5, 0, 0.8, 0.5, 0),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.1, 0, 0.1),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
    SanityCheckDataTest("a", 32, 5.0, 0, 1, 0.5, 0),
    SanityCheckDataTest("b", 32, 4.0, 1, 0.1, 0, 0.1),
    SanityCheckDataTest("a", 32, 6.0, 1, 0.9, 0.5, 0),
    SanityCheckDataTest("a", 32, 5.5, 0, 1, 0.5, 0),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.2, 0, 0.1),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.3, 0, 0.1),
    SanityCheckDataTest("a", 32, 5.0, 0, 0.6, 0.5, 0),
    SanityCheckDataTest("b", 32, 4.0, 1, 0.1, 0, 0.1),
    SanityCheckDataTest("a", 32, 6.0, 0, 0.9, 0.5, 0),
    SanityCheckDataTest("a", 32, 5.5, 0, 1, 0.5, 0),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.3, 0, 0.1),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
    SanityCheckDataTest("b", 32, 5.4, 1, 0.4, 0, 0.1)
  ).map(data =>
    (
      data.name.toText,
      data.height_null.toRealNN,
      Seq(data.age, data.height, data.isBlueEyed, data.gender, data.testFeatNegCor).toOPVector
    )
  )
  // scalastyle:on
  Spec[RecordInsightsLOCO[_]] should "work with randomly generated features and binary logistic regression" in {
    val features = RandomVector.sparse(RandomReal.normal(), 40).limit(1000)
    val labels = RandomIntegral.integrals(0, 2).limit(1000).map(_.value.get.toRealNN)
    val (df, f1, l1) = TestFeatureBuilder("features", "labels", features.zip(labels))
    val l1r = l1.copy(isResponse = true)
    val dfWithMeta = addMetaData(df, "features", 40)
    val sparkModel = new OpLogisticRegression().setInput(l1r, f1).fit(df)

    // val model = sparkModel.getSparkMlStage().get
    val insightsTransformer = new RecordInsightsLOCO(sparkModel).setInput(f1)
    val insights = insightsTransformer.transform(dfWithMeta).collect(insightsTransformer.getOutput())

    insights.foreach(_.value.size shouldBe 20)
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    parsed.map(_.count { case (_, v) => v.exists(_._1 == 1) } shouldBe 20) // number insights per pred column
    parsed.foreach(_.values.foreach(i => i.foreach(v => math.abs(v._2) > 0 shouldBe true)))
  }

  it should "work with randomly generated features and multiclass random forest" in {
    val features = RandomVector.sparse(RandomReal.normal(), 40).limit(1000)
    val labels = RandomIntegral.integrals(0, 5).limit(1000).map(_.value.get.toRealNN)
    val (df, f1, l1) = TestFeatureBuilder("features", "labels", features.zip(labels))
    val l1r = l1.copy(isResponse = true)
    val dfWithMeta = addMetaData(df, "features", 40)
    val sparkModel = new OpRandomForestClassifier().setInput(l1r, f1).fit(df)

    val insightsTransformer = new RecordInsightsLOCO(sparkModel).setInput(f1).setTopK(2)

    val insights = insightsTransformer.transform(dfWithMeta).collect(insightsTransformer.getOutput())
    insights.foreach(_.value.size shouldBe 2)
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    parsed.map(_.count { case (_, v) => v.exists(_._1 == 5) } shouldBe 0) // no 6th column of insights
    parsed.map(_.count { case (_, v) => v.exists(_._1 == 4) } shouldBe 2) // number insights per pred column
    parsed.map(_.count { case (_, v) => v.exists(_._1 == 3) } shouldBe 2) // number insights per pred column
    parsed.map(_.count { case (_, v) => v.exists(_._1 == 2) } shouldBe 2) // number insights per pred column
    parsed.map(_.count { case (_, v) => v.exists(_._1 == 1) } shouldBe 2) // number insights per pred column
    parsed.map(_.count { case (_, v) => v.exists(_._1 == 0) } shouldBe 2) // number insights per pred column
  }


  it should "work with randomly generated features and linear regression" in {
    val features = RandomVector.sparse(RandomReal.normal(), 40).limit(1000)
    val labels = RandomReal.normal[RealNN]().limit(1000)
    val (df, f1, l1) = TestFeatureBuilder("features", "labels", features.zip(labels))
    val l1r = l1.copy(isResponse = true)
    val dfWithMeta = addMetaData(df, "features", 40)
    val sparkModel = new OpLinearRegression().setInput(l1r, f1).fit(df)
    val model = sparkModel.asInstanceOf[SparkWrapperParams[_]].getSparkMlStage().get
      .asInstanceOf[LinearRegressionModel]

    val insightsTransformer = new RecordInsightsLOCO(model).setInput(f1)
    val insights = insightsTransformer.transform(dfWithMeta).collect(insightsTransformer.getOutput())
    insights.foreach(_.value.size shouldBe 20)
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    parsed.foreach(_.values.foreach(i => i.foreach(v => v._1 shouldBe 0))) // has only one pred column
    parsed.foreach(_.values.foreach(i => i.foreach(v => math.abs(v._2) > 0 shouldBe true)))
  }

  private def addMetaData(df: DataFrame, fieldName: String, size: Int): DataFrame = {
    val columns = (0 until size).map(_.toString).map(i => new OpVectorColumnMetadata(Seq(i), Seq(i), Some(i), Some(i)))
    val hist = (0 until size).map(_.toString).map(i => i -> FeatureHistory(Seq(s"a_$i"), Seq(s"b_$i"))).toMap
    val metadata = OpVectorMetadata(fieldName, columns.toArray, hist).toMetadata
    val fields = df.schema.fields.map { f =>
      if (f.name == fieldName) f.copy(metadata = metadata)
      else f
    }
    spark.createDataFrame(df.rdd, StructType(fields))
  }

  it should "return the most predictive features" in {
    val (testData, name, labelNoRes, featureVector) = TestFeatureBuilder("name", "label", "features", data)
    val label = labelNoRes.copy(isResponse = true)
    val testDataMeta = addMetaData(testData, "features", 5)
    val sparkModel = new OpLogisticRegression().setInput(label, featureVector).fit(testData)

    val transformer = new RecordInsightsLOCO(sparkModel).setInput(featureVector)

    val insights = transformer.setTopK(1).transform(testDataMeta).collect(transformer.getOutput())
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    // the highest corr that value that is not zero should be the top feature
    parsed.foreach { case in =>
      withClue(s"top features : ${in.map(_._1.columnName)}") {
        Set("3_3_3_3", "1_1_1_1").contains(in.head._1.columnName) shouldBe true
        // the scores should be the same but opposite in sign
        math.abs(in.head._2(0)._2 + in.head._2(1)._2) < 0.00001 shouldBe true
      }
    }
  }

  it should "return the most predictive features when using top K Positives + top K negatives strat" in {
    val (testData, name, labelNoRes, featureVector) = TestFeatureBuilder("name", "label", "features", data)
    val label = labelNoRes.copy(isResponse = true)
    val testDataMeta = addMetaData(testData, "features", 5)
    val sparkModel = new OpLogisticRegression().setInput(label, featureVector).fit(testData)
    val transformer = new RecordInsightsLOCO(sparkModel).setTopKStrategy(TopKStrategy.PositiveNegative)
      .setInput(featureVector)
    val insights = transformer.transform(testDataMeta)
    val parsed = insights.collect(name, transformer.getOutput())
      .map { case (n, i) => n -> RecordInsightsParser.parseInsights(i) }
    parsed.foreach { case (_, in) =>
      withClue(s"top features : ${in.map(_._1.columnName)}") {
        in.head._1.columnName == "1_1_1_1" || in.last._1.columnName == "3_3_3_3" shouldBe true
      }
    }
  }

  it should "return the most predictive features for data generated with a strong relation to the label" in {
    val numRows = 1000
    val countryData: Seq[Country] = RandomText.countries.withProbabilityOfEmpty(0.3).take(numRows).toList
    val pickListData: Seq[PickList] = RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G"))
      .withProbabilityOfEmpty(0.1).limit(numRows)
    val currencyData: Seq[Currency] = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0)
      .withProbabilityOfEmpty(0.3).limit(numRows)

    // Generate the label as a function of the features, so we know there should be strong record-level insights
    val labelData: Seq[RealNN] = pickListData.map(p =>
      p.value match {
        case Some("A") | Some("B") | Some("C") => RealNN(1.0)
        case _ => RealNN(0.0)
      }
    )

    // Generate the raw features and corresponding dataframe
    val generatedData: Seq[(Country, PickList, Currency, RealNN)] =
      countryData.zip(pickListData).zip(currencyData).zip(labelData).map {
        case (((co, pi), cu), la) => (co, pi, cu, la)
      }
    val (rawDF, rawCountry, rawPickList, rawCurrency, rawLabel) =
      TestFeatureBuilder("country", "picklist", "currency", "label", generatedData)
    val rawLabelResponse = rawLabel.copy(isResponse = true)
    val genFeatureVector = Seq(rawCountry, rawPickList, rawCurrency).transmogrify()

    // Materialize the feature vector along with the label
    val fullDF = new OpWorkflow().setResultFeatures(genFeatureVector, rawLabelResponse).transform(rawDF)

    val sparkModel = new OpRandomForestClassifier().setInput(rawLabelResponse, genFeatureVector).fit(fullDF)
    val insightsTransformer = new RecordInsightsLOCO(sparkModel).setInput(genFeatureVector).setTopK(10)
    val insights = insightsTransformer.transform(fullDF).collect(insightsTransformer.getOutput())
    val parsed = insights.map(RecordInsightsParser.parseInsights)

    // Grab the feature vector metadata for comparison against the LOCO record insights
    val vectorMeta = OpVectorMetadata(fullDF.schema.last)
    val numVectorColumns = vectorMeta.columns.length

    // Each feature vector should only have either three or four non-zero entries. One each from country and picklist,
    // while currency can have either two (if it's null since the currency column will be filled with the mean) or just
    // one if it's not null.
    parsed.length shouldBe numRows
    parsed.foreach(m => m.size <= 4 shouldBe true)

    // Want to check the average contribution strengths for each picklist response and compare them to the
    // average contribution strengths of the other features. We should have a very high contribution when choices
    // A, B, or C are present in the record (since they determine the label), and low average contributions otherwise.
    val totalImportances = parsed.foldLeft(z = Array.fill[(Double, Int)](numVectorColumns)((0.0, 0)))((res, m) => {
      m.foreach { case (k, v) => res.update(k.index, (res(k.index)._1 + v.last._2, res(k.index)._2 + 1)) }
      res
    })
    val meanImportances = totalImportances.map(x => if (x._2 > 0) x._1 / x._2 else Double.NaN)

    // Determine all the indices for insights corresponding to both the "important" and "other" features
    val nanIndices = meanImportances.zipWithIndex.filter(_._1.isNaN).map(_._2).toSet
    val abcIndices = vectorMeta.columns.filter(x => Set("A", "B", "C").contains(x.indicatorValue.getOrElse("")))
      .map(_.index).toSet -- nanIndices
    val otherIndices = vectorMeta.columns.indices.filter(x => !abcIndices.contains(x)).toSet -- nanIndices

    // Combine quantities for all the "important" features together and all the "other" features together
    val abcAvg = math.abs(abcIndices.map(meanImportances.apply).sum) / abcIndices.size
    val otherAvg = math.abs(otherIndices.map(meanImportances.apply).sum) / otherIndices.size

    // Similar calculation for the variance of each feature importance
    val varImportances = parsed.foldLeft(z = Array.fill[(Double, Int)](numVectorColumns)((0.0, 0)))((res, m) => {
      m.foreach { case (k, v) => if (abcIndices.contains(k.index)) {
        res.update(k.index, (res(k.index)._1 + math.pow(v.last._2 - abcAvg, 2), res(k.index)._2 + 1))
      } else res.update(k.index, (res(k.index)._1 + math.pow(v.last._2 - otherAvg, 2), res(k.index)._2 + 1))
      }
      res
    }).map(x => if (x._2 > 1) x._1 / x._2 else Double.NaN)
    val abcVar = math.abs(abcIndices.map(varImportances.apply).sum) / abcIndices.size
    val otherVar = math.abs(otherIndices.map(varImportances.apply).sum) / otherIndices.size

    // Strengths of features "A", "B", and "C" should be much larger the other feature strengths
    assert(abcAvg > 4 * otherAvg,
      "Average feature strengths for features involved in label formula should be " +
        "much larger than the average feature strengths of other features")
    // There should be a really large t-value when comparing the two avg feature strengths
    assert(math.abs(abcAvg - otherAvg) / math.sqrt((abcVar + otherVar) / numRows) > 10,
      "The t-value comparing the average feature strengths between important and other features should be large")

    // Record insights averaged across all records should be similar to the feature importances from Spark's RF
    val rfImportances = sparkModel.getSparkMlStage().get.featureImportances
    val abcAvgRF = abcIndices.map(rfImportances.apply).sum / abcIndices.size
    val otherAvgRF = otherIndices.map(rfImportances.apply).sum / otherIndices.size
    val avgRecordInsightRatio = math.abs(abcAvg / otherAvg)
    val featureImportanceRatio = math.abs(abcAvgRF / otherAvgRF)

    // Compare the ratio of importances between "important" and "other" features in both paradigms
    assert(math.abs(avgRecordInsightRatio - featureImportanceRatio) * 2 /
      (avgRecordInsightRatio + featureImportanceRatio) < 0.8,
      "The ratio of feature strengths between important and other features should be similar to the ratio of " +
        "feature importances from Spark's RandomForest")
  }

  for {strategy <- VectorAggregationStrategy.values} {
    it should s"aggregate values for text and textMap derived features when strategy=$strategy" in {
      val (df, featureVector, label) = generateTestTextData
      val model = new OpLogisticRegression().setInput(label, featureVector).fit(df)
      val actualInsights = generateRecordInsights(model, df, featureVector, strategy)

      withClue("TextArea can have two null indicator values") {
        actualInsights.map(p => assert(p.size == 7 || p.size == 8))
      }
      withClue("SmartTextVectorizer detects country feature as a PickList, hence no " +
        "aggregation required for LOCO on this field.") {
        actualInsights.foreach { p =>
          assert(p.keys.exists(r => r.parentFeatureOrigins == Seq(countryFeatureName) && r.indicatorValue.isDefined))
        }
      }

      assertLOCOSum(actualInsights)
      assertAggregatedText(textFeatureName, strategy, model, df, featureVector, label, actualInsights)
      assertAggregatedText(textAreaFeatureName, strategy, model, df, featureVector, label, actualInsights)
      assertAggregatedTextMap(textMapFeatureName, "k0", strategy, model, df, featureVector, label,
        actualInsights)
      assertAggregatedTextMap(textMapFeatureName, "k1", strategy, model, df, featureVector, label,
        actualInsights)
      assertAggregatedTextMap(textAreaMapFeatureName, "k0", strategy, model, df, featureVector, label,
        actualInsights)
      assertAggregatedTextMap(textAreaMapFeatureName, "k1", strategy, model, df, featureVector, label,
        actualInsights)
    }
  }

  for {strategy <- VectorAggregationStrategy.values} {
    it should "aggregate values for date, datetime, dateMap and dateTimeMap derived features when " +
      s"strategy=$strategy" in {
      val (df, featureVector, label) = generateTestDateData
      val model = new OpLogisticRegression().setInput(label, featureVector).fit(df)
      val actualInsights = generateRecordInsights(model, df, featureVector, strategy, topK = 40)

      assertLOCOSum(actualInsights)
      assertAggregatedDate(dateFeatureName, strategy, model, df, featureVector, label, actualInsights)
      assertAggregatedDate(dateTimeFeatureName, strategy, model, df, featureVector, label, actualInsights)
      assertAggregatedDateMap(dateMapFeatureName, "k0", strategy, model, df, featureVector, label,
        actualInsights)
      assertAggregatedDateMap(dateMapFeatureName, "k1", strategy, model, df, featureVector, label,
        actualInsights)
      assertAggregatedDateMap(dateTimeMapFeatureName, "k0", strategy, model, df, featureVector, label,
        actualInsights)
      assertAggregatedDateMap(dateTimeMapFeatureName, "k1", strategy, model, df, featureVector, label,
        actualInsights)
    }
  }


  private def assertLOCOSum(actualRecordInsights: Array[Map[OpVectorColumnHistory, Insights]]): Unit = {
    withClue("LOCOs sum to 0") {
      actualRecordInsights.foreach(_.values.foreach(a => assert(math.abs(a.map(_._2).sum) < 1e-10)))
    }
  }

  /**
   * Compare the aggregation made by RecordInsightsLOCO on a text field to one made manually
   *
   * @param textFeatureName Text Field Name
   */
  def assertAggregatedText(textFeatureName: String,
    strategy: VectorAggregationStrategy,
    model: OpPredictorWrapperModel[_],
    df: DataFrame,
    featureVector: FeatureLike[OPVector],
    label: FeatureLike[RealNN],
    actualInsights: Array[Map[OpVectorColumnHistory, Insights]]
  ): Unit = {
    withClue(s"Aggregate all the derived hashing tf features of rawFeature - $textFeatureName.") {
      val predicate = (history: OpVectorColumnHistory) => history.parentFeatureOrigins == Seq(textFeatureName) &&
        history.indicatorValue.isEmpty && history.descriptorValue.isEmpty
      assertAggregatedWithPredicate(predicate, strategy, model, df, featureVector, label, actualInsights)
    }
  }

  /**
   * Compare the aggregation made by RecordInsightsLOCO to one made manually
   *
   * @param textMapFeatureName Text Map Field Name
   */
  def assertAggregatedTextMap(textMapFeatureName: String, keyName: String,
    strategy: VectorAggregationStrategy,
    model: OpPredictorWrapperModel[_],
    df: DataFrame,
    featureVector: FeatureLike[OPVector],
    label: FeatureLike[RealNN],
    actualInsights: Array[Map[OpVectorColumnHistory, Insights]]
  ): Unit = {
    withClue(s"Aggregate all the derived hashing tf of rawMapFeature - $textMapFeatureName for key - $keyName") {
      val predicate = (history: OpVectorColumnHistory) => history.parentFeatureOrigins == Seq(textMapFeatureName) &&
        history.grouping == Option(keyName) && history.indicatorValue.isEmpty && history.descriptorValue.isEmpty
      assertAggregatedWithPredicate(predicate, strategy, model, df, featureVector, label, actualInsights)
    }
  }

  /**
   * Compare the aggregation made by RecordInsightsLOCO on a Date/DateTime field to one made manually
   *
   * @param dateFeatureName Date/DateTime Field
   */
  def assertAggregatedDate(dateFeatureName: String,
    strategy: VectorAggregationStrategy,
    model: OpPredictorWrapperModel[_],
    df: DataFrame,
    featureVector: FeatureLike[OPVector],
    label: FeatureLike[RealNN],
    actualInsights: Array[Map[OpVectorColumnHistory, Insights]]
  ): Unit = {
    for {timePeriod <- TransmogrifierDefaults.CircularDateRepresentations} {
      withClue(s"Aggregate x_$timePeriod and y_$timePeriod of rawFeature - $dateFeatureName.") {
        val predicate = (history: OpVectorColumnHistory) => history.parentFeatureOrigins == Seq(dateFeatureName) &&
          history.descriptorValue.isDefined &&
          history.descriptorValue.get.split("_").last == timePeriod.entryName
        assertAggregatedWithPredicate(predicate, strategy, model, df, featureVector, label, actualInsights)
      }
    }
  }

  /**
   * Compare the aggregation made by RecordInsightsLOCO on a DateMap/DateTimeMap field to one made manually
   *
   * @param dateMapFeatureName DateMap/DateTimeMap Field
   */
  def assertAggregatedDateMap(dateMapFeatureName: String, keyName: String,
    strategy: VectorAggregationStrategy,
    model: OpPredictorWrapperModel[_],
    df: DataFrame,
    featureVector: FeatureLike[OPVector],
    label: FeatureLike[RealNN],
    actualInsights: Array[Map[OpVectorColumnHistory, Insights]]
  ): Unit = {
    for {timePeriod <- TransmogrifierDefaults.CircularDateRepresentations} {
      withClue(s"Aggregate x_$timePeriod and y_$timePeriod of rawMapFeature - $dateMapFeatureName " +
        s"with key as $keyName.") {
        val predicate = (history: OpVectorColumnHistory) => history.parentFeatureOrigins == Seq(dateMapFeatureName) &&
          history.grouping == Option(keyName) && history.descriptorValue.isDefined &&
          history.descriptorValue.get.split("_").last == timePeriod.entryName
        assertAggregatedWithPredicate(predicate, strategy, model, df, featureVector, label, actualInsights)
      }
    }
  }

  /**
   * Compare the aggregation made by RecordInsightsLOCO to one made manually
   *
   * @param predicate  predicate used by RecordInsights in order to aggregate
   */
  private def assertAggregatedWithPredicate(
    predicate: OpVectorColumnHistory => Boolean,
    strategy: VectorAggregationStrategy,
    model: OpPredictorWrapperModel[_],
    df: DataFrame,
    featureVector: FeatureLike[OPVector],
    label: FeatureLike[RealNN],
    actualRecordInsights: Array[Map[OpVectorColumnHistory, Insights]]
  ): Unit = {
    implicit val enc: Encoder[(Array[Double], Long)] = ExpressionEncoder()
    implicit val enc2: Encoder[Seq[Double]] = ExpressionEncoder()

    val meta = OpVectorMetadata.apply(df.schema(featureVector.name))

    val indices = meta.getColumnHistory()
      .filter(predicate)
      .map(_.index)

    val expectedLocos = df.select(label, featureVector).map {
      case Row(l: Double, v: Vector) =>
        val featureArray = v.copy.toArray
        val baseScore = model.transformFn(l.toRealNN, v.toOPVector).score.toSeq
        strategy match {
          case VectorAggregationStrategy.Avg =>
            val locos = indices.map { i =>
              val oldVal = v(i)
              featureArray.update(i, 0.0)
              val newScore = model.transformFn(l.toRealNN, featureArray.toOPVector).score.toSeq
              featureArray.update(i, oldVal)
              baseScore.zip(newScore).map { case (b, n) => b - n }
            }
            val sumLOCOs = locos.reduce((a1, a2) => a1.zip(a2).map { case (l, r) => l + r })
            sumLOCOs.map(_ / indices.length)
          case VectorAggregationStrategy.LeaveOutVector =>
            indices.foreach { i => featureArray.update(i, 0.0) }
            val newScore = model.transformFn(l.toRealNN, featureArray.toOPVector).score.toSeq
            baseScore.zip(newScore).map { case (b, n) => b - n }
        }
    }
    val expected = expectedLocos.collect().toSeq.filter(_.head != 0.0)

    val actual = actualRecordInsights
      .flatMap(_.find { case (history, _) => predicate(history) })
      .map(_._2.map(_._2)).toSeq
    val zip = actual.zip(expected)
    zip.foreach { case (a, e) =>
      a.zip(e).foreach { case (v1, v2) => assert(math.abs(v1 - v2) < 1e-10,
        s"expected aggregated LOCO value ($v2) should be the same as actual ($v1)")
      }
    }
  }

  private def generateRecordInsights[T <: Model[T]](
    model: T,
    df: DataFrame,
    featureVector: FeatureLike[OPVector],
    strategy: VectorAggregationStrategy,
    topK: Int = 20
  ): Array[Map[OpVectorColumnHistory, Insights]] = {
    val transformer = new RecordInsightsLOCO(model).setInput(featureVector).setTopK(topK)
      .setVectorAggregationStrategy(strategy)
    val insights = transformer.transform(df)
    insights.collect(transformer.getOutput()).map(i => RecordInsightsParser.parseInsights(i))
  }
}

trait RecordInsightsTestDataGenerator extends TestSparkContext {
  self: Suite =>

  val numRows = 1000

  val labelFeatureName = "label"

  // DateFeature Names
  val dateFeatureName = "dateFeature"
  val dateTimeFeatureName = "dateTimeFeature"
  val dateMapFeatureName = "dateMapFeature"
  val dateTimeMapFeatureName = "dateTimeMapFeature"

  // TextData Feature Names
  val countryFeatureName = "country"
  val textFeatureName = "text"
  val textMapFeatureName = "textMap"
  val textAreaFeatureName = "textArea"
  val textAreaMapFeatureName = "textAreaMap"

  def generateTestDateData: (DataFrame, FeatureLike[OPVector], FeatureLike[RealNN]) = {
    val refDate = TransmogrifierDefaults.ReferenceDate.minusMillis(1)

    val minStep = 1000000
    val maxStep = 1000000000

    // Generating Data
    val dateData: Seq[Date] = RandomIntegral.dates(refDate.toDate,
      minStep, maxStep).withProbabilityOfEmpty(0.3).limit(numRows)

    val dateTimeData: Seq[DateTime] = RandomIntegral.datetimes(refDate.toDate,
      minStep, maxStep).withProbabilityOfEmpty(0.3).limit(numRows)

    val dateMapData: Seq[DateMap] = RandomMap.of(
      RandomIntegral.dates(refDate.toDate, minStep, maxStep).withProbabilityOfEmpty(0.3),
      minSize = 0, maxSize = 3
    ).limit(numRows)

    val dateTimeMapData: Seq[DateTimeMap] = RandomMap.of(
      RandomIntegral.datetimes(refDate.toDate, minStep, maxStep).withProbabilityOfEmpty(0.3),
      minSize = 0, maxSize = 3
    ).limit(numRows)

    val labelData: Seq[RealNN] = RandomIntegral.integrals(0, 2).limit(numRows).map(_.value.get.toRealNN)

    val generatedDateData: Seq[(Date, DateTime, DateMap, DateTimeMap, RealNN)] = dateData.zip(dateTimeData)
      .zip(dateMapData).zip(dateTimeMapData).zip(labelData)
      .map { case ((((d, t), dmap), tmap), l) => (d, t, dmap, tmap, l) }

    val (dateDF, date, datetime, dateMap, dateTimeMap, labelNoRes) = TestFeatureBuilder(dateFeatureName,
      dateTimeFeatureName, dateMapFeatureName, dateTimeMapFeatureName, labelFeatureName, generatedDateData)

    val rawData = dateDF.withColumn("id", monotonically_increasing_id())

    val label = labelNoRes.copy(isResponse = true)

    // Apply date vectorizer
    val dateVector = date.vectorize(dateListPivot = DateListPivot.SinceLast, referenceDate = refDate)
    val datetimeVector = datetime.vectorize(dateListPivot = DateListPivot.SinceLast, referenceDate = refDate)
    val dateMapVector = dateMap.vectorize(defaultValue = 0.0, referenceDate = refDate)
    val datetimeMapVector = dateTimeMap.vectorize(defaultValue = 0.0, referenceDate = refDate)
    val featureVector = Seq(dateVector, datetimeVector, dateMapVector, datetimeMapVector).combine()
    val featureTransformedDF = new OpWorkflow().setResultFeatures(featureVector, label).transform(rawData)

    (featureTransformedDF, featureVector, label)
  }

  def generateTestTextData: (DataFrame, FeatureLike[OPVector], FeatureLike[RealNN]) = {

    // Random Text Data
    val textData: Seq[Text] = RandomText.strings(5, 10).withProbabilityOfEmpty(0.3).take(numRows).toList

    // Random Text Area Data. Keys are k0 and k1
    val textAreaData: Seq[TextArea] = RandomText.textAreas(10, 15).withProbabilityOfEmpty(0.3).take(numRows).toList

    // Random Text Map Data. Keys are k0 and k1
    val textMapData: Seq[TextMap] = RandomMap.of(RandomText.strings(5, 10).withProbabilityOfEmpty(0.5),
      0, 3).take(numRows).toList

    // Random Text Area Map Data. Keys are k0 and k1
    val textAreaMapData: Seq[TextAreaMap] = RandomMap.of(RandomText.textAreas(5, 10).withProbabilityOfEmpty(0.5),
      0, 3).take(numRows).toList

    // Random Country Data
    val countryData: Seq[Text] = RandomText.textFromDomain(List("USA", "Mexico", "Canada")).withProbabilityOfEmpty(0.2)
      .take(numRows).toList

    // Response variable
    val labels = RandomIntegral.integrals(0, 2).limit(numRows).map(_.value.get.toRealNN)

    val generatedTextData: Seq[(Text, Text, TextMap, RealNN)] = countryData.zip(textData)
      .zip(textMapData).zip(labels).map { case (((c, t), tm), l) => (c, t, tm, l) }

    val (textDF, country, text, textMap, labelNoRes) = TestFeatureBuilder(countryFeatureName, textFeatureName,
      textMapFeatureName, labelFeatureName, generatedTextData)

    val generatedTextAreaData: Seq[(TextArea, TextAreaMap)] = textAreaData.zip(textAreaMapData)

    val (textAreaDF, textArea, textAreaMap) = TestFeatureBuilder("textArea", "textAreaMap", generatedTextAreaData)

    val textDFWithID = textDF.withColumn("id", monotonically_increasing_id())
    val textAreaDFWithID = textAreaDF.withColumn("id", monotonically_increasing_id())
    val testData = textDFWithID.join(textAreaDFWithID, "id")

    val label = labelNoRes.copy(isResponse = true)

    // Apply SmartText(Map)Vectorizer to created features
    val maxCardinality = 50
    val numHashes = 50
    val autoDetectLanguage = false
    val minTokenLength = 1
    val toLowerCase = false

    val textVectorized = text.smartVectorize(maxCardinality,
      numHashes,
      autoDetectLanguage,
      minTokenLength,
      toLowerCase,
      others = Array(country))

    val textAreaVectorized = textArea.vectorize(numHashes, autoDetectLanguage, minTokenLength, toLowerCase)

    val textAreaSmartVectorized = textArea.smartVectorize(maxCardinality, numHashes, autoDetectLanguage,
      minTokenLength, toLowerCase)

    val textMapVectorized = textMap.smartVectorize(maxCardinality,
      numHashes,
      autoDetectLanguage,
      minTokenLength,
      toLowerCase)

    val textAreaMapVectorized = textAreaMap.smartVectorize(maxCardinality,
      numHashes,
      autoDetectLanguage,
      minTokenLength,
      toLowerCase)

    val featureVector = Seq(textVectorized, textMapVectorized, textAreaVectorized, textAreaSmartVectorized,
      textAreaMapVectorized).combine()

    val vectorized = new OpWorkflow().setResultFeatures(featureVector).transform(testData)

    // Sanity Checker
    val checker = new SanityChecker().setInput(label, featureVector)

    val checkedDf = checker.fit(vectorized).transform(vectorized)

    val checkedFeatureVector = checker.getOutput()

    (checkedDf, checkedFeatureVector, label)
  }
}
