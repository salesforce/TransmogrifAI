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

package com.salesforce.op


import com.salesforce.op.utils.stages.FitStagesUtil._
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelSelector, OpLogisticRegression}
import com.salesforce.op.stages.impl.feature.{OpLDA, OpScalarStandardScaler}
import com.salesforce.op.stages.impl.preparators.SanityChecker
import com.salesforce.op.stages.impl.selector.ModelSelector
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomBinary, RandomReal, RandomVector}
import org.apache.spark.ml.{Estimator, Model}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpWorkflowCoreTest extends FlatSpec with TestSparkContext {
  // Types
  type MS = ModelSelector[_ <: Model[_], _ <: Estimator[_]]

  // Random Data
  val count = 1000
  val sizeOfVector = 100
  val seed = 1223L
  val p = 0.3
  val vectors = RandomVector.dense(RandomReal.uniform[Real](-1.0, 1.0), sizeOfVector).take(count)
  val response = RandomBinary(p).withProbabilityOfEmpty(0.0).take(count).map(_.toDouble.toRealNN(0))
  val response2 = RandomBinary(p).withProbabilityOfEmpty(0.0).take(count).map(_.toDouble.toRealNN(0))
  val (data, rawLabel, rawLabel2, features) = TestFeatureBuilder[RealNN, RealNN, OPVector]("label", "label2",
    "features", response.zip(response2).zip(vectors).map(v => (v._1._1, v._1._2, v._2)).toSeq)
  val label = rawLabel.copy(isResponse = true)
  val label2 = rawLabel2.copy(isResponse = true)

  // LDA (nonCVTS Stage)
  val lda = new OpLDA()

  // Sanity Checker (cVTS Stage)
  val sanityChecker = new SanityChecker()

  // Workflow
  val wf = new OpWorkflow()

  Spec[OpWorkflowCore] should "handle empty DAG" in {
    assert(
      res = cutDAG(wf),
      expected = CutDAG(
        modelSelector = None,
        before = Array.empty[Layer],
        during = Array.empty[Layer],
        after = Array.empty[Layer]
      )
    )
  }

  it should "cut simple DAG containing modelSelector only" in {
    val ms = BinaryClassificationModelSelector()
    val pred = ms.setInput(label, features).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(pred)),
      expected = CutDAG(
        modelSelector = Option((ms, 0)),
        before = Array.empty[Layer],
        during = Array.empty[Layer],
        after = Array.empty[Layer]
      )
    )
  }

  it should "cut simple DAG with nonCVTS and cVTS stage" in {
    val ldaFeatures = lda.setInput(features).getOutput()
    val checkedFeatures = sanityChecker.setInput(label, ldaFeatures).getOutput()
    val ms = BinaryClassificationModelSelector()
    val pred = ms.setInput(label, checkedFeatures).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(pred)),
      expected = CutDAG(
        modelSelector = Option((ms, 0)),
        before = Array(Array((lda, 2))),
        during = Array(Array((sanityChecker, 1))),
        after = Array.empty[Layer]
      )
    )
  }

  it should "cut simple DAG with nonCVTS and cVTS stage and stages after CV" in {
    val ldaFeatures = lda.setInput(features).getOutput()
    val checkedFeatures = sanityChecker.setInput(label, ldaFeatures).getOutput()
    val ms = BinaryClassificationModelSelector()
    val pred = ms.setInput(label, checkedFeatures).getOutput()
    val predValue = pred.map[RealNN](_.prediction.toRealNN)
    val zNormalize = new OpScalarStandardScaler()
    val realPred = zNormalize.setInput(predValue).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(realPred)),
      expected = CutDAG(
        modelSelector = Option((ms, 2)),
        before = Array(Array((lda, 4))),
        during = Array(Array((sanityChecker, 3))),
        after = Array(Array((predValue.originStage, 1)), Array((zNormalize, 0)))
      )
    )
  }

  it should "cut DAG with no nonCVTS stage" in {
    val checkedFeatures = sanityChecker.setInput(label, features).getOutput()
    val ms = BinaryClassificationModelSelector()
    val pred = ms.setInput(label, checkedFeatures).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(pred)),
      expected = CutDAG(
        modelSelector = Option((ms, 0)),
        before = Array.empty[Layer],
        during = Array(Array((sanityChecker, 1))),
        after = Array.empty[Layer]
      )
    )
  }

  it should "cut DAG with no cVTS stage before ModelSelector" in {
    val ms = BinaryClassificationModelSelector()
    val ldaFeatures = lda.setInput(features).getOutput()
    val pred = ms.setInput(label, ldaFeatures).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(pred)),
      expected = CutDAG(
        modelSelector = Option((ms, 0)),
        before = Array(Array((lda, 1))),
        during = Array.empty[Layer],
        after = Array.empty[Layer]
      )
    )
  }

  it should "cut DAG with no ModelSelector" in {
    val ldaFeatures = lda.setInput(features).getOutput()
    val checkedFeatures = sanityChecker.setInput(label, ldaFeatures).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(checkedFeatures)),
      expected = CutDAG(
        modelSelector = None,
        before = Array.empty[Layer],
        during = Array.empty[Layer],
        after = Array.empty[Layer]
      )
    )
  }

  it should "throw an error when there is more than one ModelSelector in parallel" in {
    val ms1 = BinaryClassificationModelSelector()
    val ms2 = BinaryClassificationModelSelector()
    val pred1 = ms1.setInput(label, features).getOutput()
    val pred2 = ms2.setInput(label2, features).getOutput()

    val error = intercept[IllegalArgumentException](cutDAG(wf.setResultFeatures(pred1, pred2)))
    error.getMessage
      .contains(s"OpWorkflow can contain at most 1 Model Selector. Found 2 Model Selectors :") shouldBe true
  }

  it should "throw an error when there is more than one ModelSelector in sequence" in {
    val ms1 = BinaryClassificationModelSelector()
    val ms2 = BinaryClassificationModelSelector()
    val pred1 = ms1.setInput(label, features).getOutput()
    val predLabel = pred1.map[RealNN](_.prediction.toRealNN)
    val pred2 = ms2.setInput(predLabel, features).getOutput()

    val error = intercept[IllegalArgumentException](cutDAG(wf.setResultFeatures(pred1, pred2)))
    error.getMessage
      .contains(s"OpWorkflow can contain at most 1 Model Selector. Found 2 Model Selectors :") shouldBe true
  }

  it should "optimize the DAG by removing stages no not related to model selection" in {
    val ms = BinaryClassificationModelSelector()
    val logReg = new OpLogisticRegression()
    val ldaFeatures = lda.setInput(features).getOutput()
    val checkedFeatures = sanityChecker.setInput(label2, ldaFeatures).getOutput()
    val pred = ms.setInput(label, features).getOutput()
    val predLogReg = logReg.setInput(label2, checkedFeatures).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(pred, predLogReg)),
      expected = CutDAG(
        modelSelector = Option((ms, 0)),
        before = Array(Array((lda, 2)), Array((sanityChecker, 1)), Array((logReg, 0))),
        during = Array.empty[Layer],
        after = Array.empty[Layer]
      )
    )
  }

  it should "cut simple DAG without taking label transformation as cVTS stage" in {
    val ldaFeatures = lda.setInput(features).getOutput()
    val zNormalize = new OpScalarStandardScaler()
    val transformedLabel: FeatureLike[RealNN] = zNormalize.setInput(label).getOutput()
    val checkedFeatures = sanityChecker.setInput(transformedLabel, ldaFeatures).getOutput()
    val ms = BinaryClassificationModelSelector()
    val pred = ms.setInput(transformedLabel, checkedFeatures).getOutput()

    assert(
      res = cutDAG(wf.setResultFeatures(pred)),
      expected = CutDAG(
        modelSelector = Option((ms, 0)),
        before = Array(Array((lda, 2), (zNormalize, 2))),
        during = Array(Array((sanityChecker, 1))),
        after = Array.empty[Layer]
      )
    )
  }

  /**
   * Compare Actual and expected cut DAGs
   *
   * @param res      actual cut
   * @param expected expected cut
   */
  private def assert(res: CutDAG, expected: CutDAG): Unit = {
    res.modelSelector shouldBe expected.modelSelector
    res.before should contain theSameElementsInOrderAs expected.before
    res.during should contain theSameElementsInOrderAs expected.during
    res.after should contain theSameElementsInOrderAs expected.after
  }
}


