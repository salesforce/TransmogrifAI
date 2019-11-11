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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SmartTextVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[Text, OPVector], SmartTextVectorizer[Text]] with AttributeAsserts {

  lazy val (inputData, f1, f2) = TestFeatureBuilder("text1", "text2",
    Seq[(Text, Text)](
      ("hello world".toText, "Hello world!".toText),
      ("hello world".toText, "What's up".toText),
      ("good evening".toText, "How are you doing, my friend?".toText),
      ("hello world".toText, "Not bad, my friend.".toText),
      (Text.empty, Text.empty)
    )
  )
  val estimator: SmartTextVectorizer[Text] = new SmartTextVectorizer()
    .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
    .setTopK(2).setPrependFeatureName(false)
    .setHashSpaceStrategy(HashSpaceStrategy.Shared)
    .setInput(f1, f2)

  val expectedResult: Seq[OPVector] = Seq(
    Vectors.sparse(9, Array(0, 4, 6), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(9, Array(0, 8), Array(1.0, 1.0)),
    Vectors.sparse(9, Array(1, 6), Array(1.0, 1.0)),
    Vectors.sparse(9, Array(0, 6), Array(1.0, 2.0)),
    Vectors.sparse(9, Array(3, 8), Array(1.0, 1.0))
  ).map(_.toOPVector)

  it should "detect one categorical and one non-categorical text feature" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val categoricalVectorized = new OpTextPivotVectorizer[Text]().setMinSupport(1).setTopK(2).setInput(f1).getOutput()
    val tokenizedText = new TextTokenizer[Text]().setInput(f2).getOutput()
    val textVectorized = new OPCollectionHashingVectorizer[TextList]()
      .setNumFeatures(4).setPrependFeatureName(false).setInput(tokenizedText).getOutput()
    val nullIndicator = new TextListNullTransformer[TextList]().setInput(tokenizedText).getOutput()

    val transformed = new OpWorkflow()
      .setResultFeatures(smartVectorized, categoricalVectorized, textVectorized, nullIndicator).transform(inputData)
    val result = transformed.collect(smartVectorized, categoricalVectorized, textVectorized, nullIndicator)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true, transformed.collect(smartVectorized))
    val fieldCategorical = transformed.schema(categoricalVectorized.name)
    val catRes = transformed.collect(categoricalVectorized)
    assertNominal(fieldCategorical, Array.fill(catRes.head.value.size)(true), catRes)
    val fieldText = transformed.schema(textVectorized.name)
    val textRes = transformed.collect(textVectorized)
    assertNominal(fieldText, Array.fill(textRes.head.value.size)(false), textRes)
    val (smart, expected) = result.map { case (smartVector, categoricalVector, textVector, nullVector) =>
      val combined = categoricalVector.combine(textVector, nullVector)
      smartVector -> combined
    }.unzip

    smart shouldBe expected
  }

  it should "detect two categorical text features" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(10).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val categoricalVectorized =
      new OpTextPivotVectorizer[Text]().setMinSupport(1).setTopK(2).setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized, categoricalVectorized).transform(inputData)
    val result = transformed.collect(smartVectorized, categoricalVectorized)
    val field = transformed.schema(smartVectorized.name)
    val smartRes = transformed.collect(smartVectorized)
    assertNominal(field, Array.fill(smartRes.head.value.size)(true), smartRes)
    val fieldCategorical = transformed.schema(categoricalVectorized.name)
    val catRes = transformed.collect(categoricalVectorized)
    assertNominal(fieldCategorical, Array.fill(catRes.head.value.size)(true), catRes)
    val (smart, expected) = result.unzip

    smart shouldBe expected
  }

  it should "detect two non categorical text features" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val f1Tokenized = new TextTokenizer[Text]().setInput(f1).getOutput()
    val f2Tokenized = new TextTokenizer[Text]().setInput(f2).getOutput()
    val textVectorized = new OPCollectionHashingVectorizer[TextList]()
      .setNumFeatures(4).setPrependFeatureName(false).setInput(f1Tokenized, f2Tokenized).getOutput()
    val nullIndicator = new TextListNullTransformer[TextList]().setInput(f1Tokenized, f2Tokenized).getOutput()

    val transformed = new OpWorkflow()
      .setResultFeatures(smartVectorized, textVectorized, nullIndicator).transform(inputData)
    val result = transformed.collect(smartVectorized, textVectorized, nullIndicator)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(8)(false) ++ Array(true, true), transformed.collect(smartVectorized))
    val fieldText = transformed.schema(textVectorized.name)
    val textRes = transformed.collect(textVectorized)
    assertNominal(fieldText, Array.fill(textRes.head.value.size)(false), textRes)
    val (smart, expected) = result.map { case (smartVector, textVector, nullVector) =>
      val combined = textVector.combine(nullVector)
      smartVector -> combined
    }.unzip

    smart shouldBe expected
  }

  it should "work for shortcut" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setInput(f1, f2).getOutput()

    val shortcutVectorized = f1.smartVectorize(
      maxCategoricalCardinality = 2, numHashes = 4, minSupport = 1, topK = 2,
      autoDetectLanguage = false, minTokenLength = 1, toLowercase = false,
      hashSpaceStrategy = HashSpaceStrategy.Shared, others = Array(f2)
    )

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized, shortcutVectorized).transform(inputData)
    val result = transformed.collect(smartVectorized, shortcutVectorized)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true, transformed.collect(smartVectorized))
    val fieldShortcut = transformed.schema(shortcutVectorized.name)
    assertNominal(fieldShortcut, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true,
      transformed.collect(shortcutVectorized))
    val (regular, shortcut) = result.unzip

    regular shouldBe shortcut
  }

  it should "fail with an error" in {
    val emptyDF = inputData.filter(inputData("text1") === "").toDF()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val thrown = intercept[IllegalArgumentException] {
      new OpWorkflow().setResultFeatures(smartVectorized).transform(emptyDF)
    }
    assert(thrown.getMessage.contains("requirement failed"))
  }

  it should "generate metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 9
    meta.columns.foreach { col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "generate categorical metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(4).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 8
    meta.columns.foreach { col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
      } else if (col.index == 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "generate non categorical metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 10
    meta.columns.foreach { col =>
      if (col.index < 4) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe None
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else if (col.index == 8) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "append the text lengths to the feature vector if one feature is determined to be text" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setTrackTextLen(true).setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)
    val res = transformed.collect(smartVectorized)
    val expected = Array(
      Vectors.sparse(10, Array(0, 4, 6, 8), Array(1.0, 1.0, 1.0, 10.0)),
      Vectors.sparse(10, Array(0, 9), Array(1.0, 1.0)),
      Vectors.sparse(10, Array(1, 6, 8), Array(1.0, 1.0, 6.0)),
      Vectors.sparse(10, Array(0, 6, 8), Array(1.0, 2.0, 9.0)),
      Vectors.sparse(10, Array(3, 9), Array(1.0, 1.0))
    ).map(_.toOPVector)

    res shouldEqual expected

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 10
    meta.columns.foreach { col =>
      if (col.index < 4) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else if (col.index == 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe None
        col.descriptorValue shouldBe Option(OpVectorColumnMetadata.TextLenString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "append one text lengths column to the feature vector for each feature determined to be text" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setTrackTextLen(true).setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)
    val res = transformed.collect(smartVectorized)
    val expected = Array(
      Vectors.sparse(12, Array(0, 2, 4, 6, 8, 9), Array(1.0, 1.0, 1.0, 1.0, 10.0, 10.0)),
      Vectors.sparse(12, Array(0, 2, 8, 11), Array(1.0, 1.0, 10.0, 1.0)),
      Vectors.sparse(12, Array(0, 3, 6, 8, 9), Array(1.0, 1.0, 1.0, 11.0, 6.0)),
      Vectors.sparse(12, Array(0, 2, 6, 8, 9), Array(1.0, 1.0, 2.0, 10.0, 9.0)),
      Vectors.sparse(12, Array(10, 11), Array(1.0, 1.0))
    ).map(_.toOPVector)

    res shouldEqual expected

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 12

    meta.columns(1)

    meta.columns.foreach { col =>
      if (col.index < 4) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe None
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else if (col.index == 8) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe None
        col.descriptorValue shouldBe Option(OpVectorColumnMetadata.TextLenString)
      } else if (col.index == 9) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe None
        col.descriptorValue shouldBe Option(OpVectorColumnMetadata.TextLenString)
      } else if (col.index == 10) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  /* TESTS FOR DETECTING SENSITIVE FEATURES BEGIN */
  val biasEstimator: SmartTextVectorizer[Text] = new SmartTextVectorizer()
    .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
    .setTopK(2).setPrependFeatureName(false)
    .setHashSpaceStrategy(HashSpaceStrategy.Shared)
    .setSensitiveFeatureMode(SensitiveFeatureMode.DetectAndRemove)
    .setInput(f1, f2)

  lazy val (newInputData, newF1, newF2, newF3) = TestFeatureBuilder("text1", "text2", "name",
    Seq[(Text, Text, Text)](
      ("hello world".toText, "Hello world!".toText, "Michael".toText),
      ("hello world".toText, "What's up".toText, "Michelle".toText),
      ("good evening".toText, "How are you doing, my friend?".toText, "Roxanne".toText),
      ("hello world".toText, "Not bad, my friend.".toText, "Ross".toText),
      (Text.empty, Text.empty, Text.empty)
    )
  )

  it should "detect a single name feature and return empty vectors" in {
    val newEstimator: SmartTextVectorizer[Text] = biasEstimator.setInput(newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    model.args.isName shouldBe Array(x = true)

    val smartVectorized = newEstimator.getOutput()

    val transformed = new OpWorkflow()
      .setResultFeatures(smartVectorized).transform(newInputData)
    val result = transformed.collect(smartVectorized)

    val (smart, expected) = result.map(smartVector => smartVector -> OPVector.empty).unzip

    smart shouldBe expected
  }

  it should "detect a single name column among other non-name Text columns" in {
    val newEstimator: SmartTextVectorizer[Text] = biasEstimator.setInput(newF1, newF2, newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    model.args.isName shouldBe Array(false, false, true)
  }

  it should "compute gender probabilities for one column in the metadata" in {
    val newEstimator: SmartTextVectorizer[Text] = biasEstimator.setInput(newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    newEstimator.getMetadata().getDoubleArray("bestFirstNameIndexes") shouldBe Array(0.0)
    newEstimator.getMetadata().getDoubleArray("pctMale") shouldBe Array(0.4)
  }

  it should "compute gender probabilities for two columns in the metadata" in {
    val newEstimator: SmartTextVectorizer[Text] = biasEstimator.setInput(newF2, newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    newEstimator.getMetadata().getDoubleArray("bestFirstNameIndexes").length shouldBe 2
    newEstimator.getMetadata().getDoubleArray("pctMale").length shouldBe 2
    newEstimator.getMetadata().getDoubleArray("pctMale").last shouldBe 0.4
  }

  it should "compute gender probabilities for three columns in the metadata" in {
    val newEstimator: SmartTextVectorizer[Text] = biasEstimator.setInput(newF1, newF2, newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    newEstimator.getMetadata().getDoubleArray("bestFirstNameIndexes").length shouldBe 3
    newEstimator.getMetadata().getDoubleArray("pctMale").length shouldBe 3
    newEstimator.getMetadata().getDoubleArray("pctMale").last shouldBe 0.4
  }

  it should "compute the same number of name check results as the number of feature columns" in {
    val numFeatures = 10
    val (ds, untypedFeatures) = TestFeatureBuilder(
      Seq.fill[Seq[Text]](numFeatures)(Seq(Text("TESTSTRING"))): _*
    )
    val features = untypedFeatures.map(_.asInstanceOf[Feature[Text]])
    for {i <- 4 to numFeatures} {
      val newEstimator: SmartTextVectorizer[Text] = biasEstimator.setInput(features.slice(0, i): _*)
      val model: SmartTextVectorizerModel[Text] = newEstimator
        .fit(ds)
        .asInstanceOf[SmartTextVectorizerModel[Text]]
      newEstimator.guardCheckResults match {
        case Some(results) => results.length shouldBe i
        case None => fail("Guard check results were not generated")
      }
      newEstimator.getMetadata().getBooleanArray("treatAsName").length shouldBe i
      newEstimator.getMetadata().getDoubleArray("predictedNameProb").length shouldBe i
      newEstimator.getMetadata().getDoubleArray("bestFirstNameIndexes").length shouldBe i
      newEstimator.getMetadata().getDoubleArray("pctMale").length shouldBe i
      newEstimator.getMetadata().getDoubleArray("pctFemale").length shouldBe i
      newEstimator.getMetadata().getDoubleArray("pctOther").length shouldBe i
    }
  }

  it should "not identify a single repeated name as Name" in {
    val (newNewInputData, newNewF1, newNewF2) = TestFeatureBuilder("repeatedname", "names",
      Seq.fill(200)("Michael").toText zip
        RandomText.names.withProbabilityOfEmpty(0.0).take(200).toSeq.map(_.asInstanceOf[Text])
    )
    val newEstimator: SmartTextVectorizer[Text] = biasEstimator.setInput(newNewF1, newNewF2)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newNewInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newNewInputData.show()
    model.args.isName shouldBe Array(false, true)
  }
  /* TESTS FOR DETECTING SENSITIVE FEATURES END */

  Spec[TextStats] should "aggregate correctly" in {
    val l1 = TextStats(Map("hello" -> 1, "world" -> 2))
    val r1 = TextStats(Map("hello" -> 1, "world" -> 1))
    val expected1 = TextStats(Map("hello" -> 2, "world" -> 3))

    val l2 = TextStats(Map("hello" -> 1, "world" -> 2, "ocean" -> 3))
    val r2 = TextStats(Map("hello" -> 1))
    val expected2 = TextStats(Map("hello" -> 1, "world" -> 2, "ocean" -> 3))

    TextStats.monoid(2).plus(l1, r1) shouldBe expected1
    TextStats.monoid(2).plus(l2, r2) shouldBe expected2
  }
}
