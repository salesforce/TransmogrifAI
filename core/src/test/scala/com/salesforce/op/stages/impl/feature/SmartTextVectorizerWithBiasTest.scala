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

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.TestFeatureBuilder
import com.salesforce.op.testkit.RandomText

class SmartTextVectorizerWithBiasTest extends SmartTextVectorizerTest {
  override val estimator: SmartTextVectorizerWithBias[Text] = new SmartTextVectorizerWithBias()
    .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
    .setTopK(2).setPrependFeatureName(false)
    .setHashSpaceStrategy(HashSpaceStrategy.Shared)
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

  // TODO: Return empty vectors for identified name features
  it should "detect a single name feature and return empty vectors" in {
    val newEstimator: SmartTextVectorizerWithBias[Text] = estimator.setInput(newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    model.args.isName shouldBe Array(true)
  }

  it should "detect a single name column among other non-name Text columns" in {
    val newEstimator: SmartTextVectorizerWithBias[Text] = estimator.setInput(newF1, newF2, newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    model.args.isName shouldBe Array(false, false, true)
  }

  it should "compute gender probabilities for one column in the metadata" in {
    val newEstimator: SmartTextVectorizerWithBias[Text] = estimator.setInput(newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    newEstimator.getMetadata().getDoubleArray("bestIndexes") shouldBe Array(0.0)
    newEstimator.getMetadata().getDoubleArray("pctMale") shouldBe Array(0.4)
  }

  it should "compute gender probabilities for two columns in the metadata" in {
    val newEstimator: SmartTextVectorizerWithBias[Text] = estimator.setInput(newF2, newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    newEstimator.getMetadata().getDoubleArray("bestIndexes").length shouldBe 2
    newEstimator.getMetadata().getDoubleArray("pctMale").length shouldBe 2
    newEstimator.getMetadata().getDoubleArray("pctMale").last shouldBe 0.4
  }

  it should "compute gender probabilities for three columns in the metadata" in {
    val newEstimator: SmartTextVectorizerWithBias[Text] = estimator.setInput(newF1, newF2, newF3)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newInputData.show()
    newEstimator.getMetadata().getDoubleArray("bestIndexes").length shouldBe 3
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
      val newEstimator: SmartTextVectorizerWithBias[Text] = estimator.setInput(features.slice(0, i): _*)
      val model: SmartTextVectorizerModel[Text] = newEstimator
        .fit(ds)
        .asInstanceOf[SmartTextVectorizerModel[Text]]
      newEstimator.guardCheckResults match {
        case Some(results) => results.length shouldBe i
        case None => fail("Guard check results were not generated")
      }
      newEstimator.getMetadata().getBooleanArray("treatAsName").length shouldBe i
      newEstimator.getMetadata().getDoubleArray("predictedNameProb").length shouldBe i
      newEstimator.getMetadata().getDoubleArray("bestIndexes").length shouldBe i
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
    val newEstimator: SmartTextVectorizerWithBias[Text] = estimator.setInput(newNewF1, newNewF2)
    val model: SmartTextVectorizerModel[Text] = newEstimator
      .fit(newNewInputData)
      .asInstanceOf[SmartTextVectorizerModel[Text]]
    newNewInputData.show()
    model.args.isName shouldBe Array(false, true)
  }
}
