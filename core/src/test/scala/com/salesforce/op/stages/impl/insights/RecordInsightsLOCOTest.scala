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

import com.salesforce.op.{FeatureHistory, OpWorkflow}
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op._
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.stages.impl.preparators.{SanityCheckDataTest, SanityChecker}
import com.salesforce.op.stages.impl.regression.OpLinearRegression
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomIntegral, RandomMap, RandomReal, RandomText, RandomVector}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnHistory, OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Encoder, Row}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.linalg._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class RecordInsightsLOCOTest extends FlatSpec with TestSparkContext {

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

  it should "aggregate LOCOS for text and textMap derived features" in {

    // Generating Data
    val numRows = 1000
    // Random Text Data
    val textData: Seq[Text] = RandomText.strings(5, 10).withProbabilityOfEmpty(0.3).take(numRows).toList
    // Random Text Map Data. Keys are k0 and k1
    val textMapData: Seq[TextMap] = RandomMap.of(RandomText.strings(5, 10).withProbabilityOfEmpty(0.5),
      0, 3).take(numRows).toList
    // Random Country Data
    val countryData: Seq[Text] = RandomText.textFromDomain(List("USA", "Mexico", "Canada")).withProbabilityOfEmpty(0.2)
      .take(numRows).toList
    // Response variable
    val labels = RandomIntegral.integrals(0, 2).limit(numRows).map(_.value.get.toRealNN)


    val generatedData: Seq[(Text, Text, TextMap, RealNN)] = countryData.zip(textData).zip(textMapData).zip(labels).map {
      case (((c, t), t2), l) => (c, t, t2, l)
    }

    val (testData, country, text, textMap, labelNoRes) = TestFeatureBuilder("country", "text", "textMap", "label",
      generatedData)
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

    val textMapVectorized = textMap.smartVectorize(maxCardinality,
      numHashes,
      autoDetectLanguage,
      minTokenLength,
      toLowerCase)

    val featureVector = Seq(textVectorized, textMapVectorized).combine()

    val vectorized = new OpWorkflow().setResultFeatures(featureVector).transform(testData)

    // Sanity Checker
    val checker = new SanityChecker().setInput(label, featureVector)

    val checked = checker.fit(vectorized).transform(vectorized)

    val checkedFeatureVector = checker.getOutput()

    // RecordInsightsLOCO
    val sparkModel = new OpLogisticRegression().setInput(label, checkedFeatureVector).fit(checked)

    val transformer = new RecordInsightsLOCO(sparkModel).setInput(checkedFeatureVector)

    val insights = transformer.transform(checked)

    val parsed = insights.collect(transformer.getOutput()).map(i => RecordInsightsParser.parseInsights(i))

    parsed.map(_.size shouldBe 4)
    parsed.foreach(p => assert(p.keys.exists(r => r.parentFeatureOrigins == Seq(country.name)
      && r.indicatorValue.isDefined), "SmartVectorizer detects country feature as a PickList, hence no " +
      "aggregation required for LOCO on this field."))
    parsed.foreach(_.values.foreach(a => assert(math.abs(a.map(_._2).sum) < 1e-10, "LOCOs sum to 0")))

    val meta = OpVectorMetadata.apply(checked.schema(checkedFeatureVector.name))

    implicit val enc: Encoder[(Array[Double], Long)] = ExpressionEncoder()
    implicit val enc2: Encoder[Seq[Double]] = ExpressionEncoder()

    /**
     * Compare the aggregation made by RecordInsightsLOCO to one made manually
     *
     * @param textFeature Text(Map) Field
     * @param predicate   predicate used by RecordInsights in order to aggregate
     */
    def assertAggregatedWithPredicate(
      textFeature: FeatureLike[_],
      predicate: OpVectorColumnHistory => Boolean
    ): Unit = {
      val textIndices = meta.getColumnHistory()
        .filter(c => predicate(c) && c.indicatorValue.isEmpty && c.descriptorValue.isEmpty)
        .map(_.index)

      val expectedLocos = checked.select(label, checkedFeatureVector).map { case Row(l: Double, v: Vector) =>
        val featureArray = v.toArray
        textIndices.map { i =>
          val oldVal = v(i)
          val baseScore = sparkModel.transformFn(l.toRealNN, v.toOPVector).score
          featureArray.update(i, 0.0)
          val newScore = sparkModel.transformFn(l.toRealNN, featureArray.toOPVector).score
          featureArray.update(i, oldVal)
          baseScore.zip(newScore).map { case (b, n) => b - n } -> 1L
        }.reduce((a, b) => a._1.zip(b._1).map { case (v1, v2) => v1 + v2 } -> (a._2 + b._2))
      }.map { case (a: Array[Double], n: Long) => a.map(_ / n).toSeq }
      val expected = expectedLocos.collect().toSeq.filter(_.head != 0.0)

      val actual = parsed.map(_.find { case (history, _) => predicate(history) }.get)
        .filter(_._1.indicatorValue.isEmpty).map(_._2.map(_._2)).toSeq
      val zip = actual.zip(expected)
      zip.foreach { case (a, e) =>
        a.zip(e).foreach { case (v1, v2) => assert(math.abs(v1 - v2) < 1e-10,
          s"expected aggregated LOCO ($v2) should be the same as actual ($v1)")
        }
      }
    }

    /**
     * Compare the aggregation made by RecordInsightsLOCO on a text field to one made manually
     *
     * @param textFeature Text Field
     */
    def assertAggregatedText(textFeature: FeatureLike[Text]): Unit = {
      val predicate = (history: OpVectorColumnHistory) => history.parentFeatureOrigins == Seq(textFeature.name)
      assertAggregatedWithPredicate(textFeature, predicate)
    }

    /**
     * Compare the aggregation made by RecordInsightsLOCO to one made manually
     *
     * @param textMapFeature Text Map Field
     */
    def assertAggregatedTextMap(textMapFeature: FeatureLike[TextMap], keyName: String): Unit = {
      val predicate = (history: OpVectorColumnHistory) => history.parentFeatureOrigins == Seq(textMapFeature.name) &&
        history.grouping == Option(keyName)
      assertAggregatedWithPredicate(textMapFeature, predicate)
    }

    assertAggregatedText(text)
    assertAggregatedTextMap(textMap, "k0")
    assertAggregatedTextMap(textMap, "k1")


  }
}
