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

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.ProbabilisticClassifierType.{ProbClassifier, ProbClassifierModel}
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomBinary, RandomIntegral, RandomReal, RandomVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.MetadataBuilder
import com.salesforce.op.utils.spark.RichDataset._

@RunWith(classOf[JUnitRunner])
class OpValidatorTest extends FlatSpec with TestSparkContext {
  // Random Data
  val count = 1000
  val sizeOfVector = 2
  val seed = 12345L
  val p = 0.325
  val multiClassProbabilities = Array(0.21, 0.29, 0.5)
  val vectors = RandomVector.sparse(RandomReal.uniform[Real](-1.0, 1.0), sizeOfVector).take(count)
  val response = RandomBinary(p).withProbabilityOfEmpty(0.0).take(count).map(_.toDouble.toRealNN(0.0))
  val multiResponse = multiClassProbabilities.zipWithIndex
    .flatMap { case (p, index) => RandomIntegral.integrals(index, index + 1).withProbabilityOfEmpty(0.0)
      .take((p * count).toInt).map(_.toDouble.toRealNN(0.0))
    }.toIterator
  val (data, rawLabel, features, rawMultiLabel) = TestFeatureBuilder[RealNN, OPVector, RealNN]("label",
    "features", "multiLabel", response.zip(vectors).zip(multiResponse)
      .map { case ((l, f), multiL) => (l, f, multiL) }.toSeq)
  val label = rawLabel.copy(isResponse = true)
  val multiLabel = rawMultiLabel.copy(isResponse = true)

  val cv = new OpCrossValidation[ProbClassifierModel, ProbClassifier](evaluator = Evaluators.BinaryClassification(),
    seed = seed, stratify = true)

  val ts = new OpTrainValidationSplit[ProbClassifierModel, ProbClassifier](
    evaluator = Evaluators.BinaryClassification(),
    seed = seed,
    stratify = true
  )

  val binaryDS = data.select(label, features)
  val multiDS = data.select(multiLabel, features)

  val condition = cv.isClassification && cv.stratify
  val balancer = Option(new DataBalancer())
  val cutter = Option(new DataCutter())


  Spec[OpCrossValidation[_, _]] should "stratify binary class data" in {
    val splits = cv.createTrainValidationSplits(condition, binaryDS, label.name, balancer)
    splits.length shouldBe ValidatorParamDefaults.NumFolds
    splits.foreach { case (train, validate) =>
      assertFractions(Array(1 - p, p), train)
      assertFractions(Array(1 - p, p), validate)
    }
    balancer.get.summary.get.toMetadata() should not be new MetadataBuilder().build()
  }

  it should "stratify multi class data" in {
    val splits = cv.createTrainValidationSplits(condition, multiDS, multiLabel.name, cutter)
    splits.length shouldBe ValidatorParamDefaults.NumFolds
    splits.foreach { case (train, validate) =>
      assertFractions(multiClassProbabilities, train)
      assertFractions(multiClassProbabilities, validate)
    }
  }


  Spec[OpTrainValidationSplit[_, _]] should "stratify binary class data" in {
    val splits = ts.createTrainValidationSplits(condition, binaryDS, label.name, balancer)
    splits.foreach { case (train, validate) =>
      assertFractions(Array(1 - p, p), train)
      assertFractions(Array(1 - p, p), validate)
    }
    balancer.get.summary.get.toMetadata() should not be new MetadataBuilder().build()
  }

  it should "stratify multi class data" in {
    val splits = ts.createTrainValidationSplits(condition, multiDS, multiLabel.name, cutter)
    splits.foreach { case (train, validate) =>
      assertFractions(multiClassProbabilities, train)
      assertFractions(multiClassProbabilities, validate)
    }
  }

  /**
   * Assert Fractions in Stratified data
   *
   * @param fractions Expected proportions
   * @param rdd       Actual Data
   */
  private def assertFractions(fractions: Array[Double], rdd: RDD[Row]): Unit = {
    val n: Double = rdd.count()
    val fractionsByClass = rdd.map { case Row(label: Double, feature: Vector) =>
      label -> (feature, label)
    }.groupByKey().mapValues(_.size / n).sortBy(_._1).values.collect()

    fractions zip fractionsByClass map { case (expected, actual) =>
      math.abs(expected - actual) should be < 0.065
    }
  }

}
