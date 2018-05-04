/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.ProbabilisticClassifierType.{ProbClassifier, ProbClassifierModel}
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomBinary, RandomIntegral, RandomReal, RandomVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.apache.spark.ml.linalg.Vector

@RunWith(classOf[JUnitRunner])
class OpValidatorTest extends FlatSpec with TestSparkContext {
  // Random Data
  val count = 1000
  val sizeOfVector = 2
  val seed = 1234L
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

  val rdd = data.withColumn(ModelSelectorBaseNames.idColName, monotonically_increasing_id()).rdd

  val binaryRDD = rdd.map {
    case Row(label, features, _, index) => (label, features, index).asInstanceOf[LabelFeaturesKey]
  }

  val multiRDD = rdd.map {
    case Row(_, features, multiLabel, index) => (multiLabel, features, index).asInstanceOf[LabelFeaturesKey]
  }


  Spec[OpCrossValidation[_, _]] should "stratify binary class data" in {
    val splits = cv.createTrainValidationSplits(binaryRDD)
    splits.foreach { case (train, validate) =>
      assertFractions(Array(1 - p, p), train)
      assertFractions(Array(1 - p, p), validate)
    }
  }

  it should "stratify multi class data" in {
    val splits = cv.createTrainValidationSplits(multiRDD)
    splits.foreach { case (train, validate) =>
      assertFractions(multiClassProbabilities, train)
      assertFractions(multiClassProbabilities, validate)
    }
  }


  Spec[OpTrainValidationSplit[_, _]] should "stratify binary class data" in {
    val splits = ts.createTrainValidationSplits(binaryRDD)
    splits.foreach { case (train, validate) =>
      assertFractions(Array(1 - p, p), train)
      assertFractions(Array(1 - p, p), validate)
    }
  }

  it should "stratify multi class data" in {
    val splits = ts.createTrainValidationSplits(multiRDD)
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
      math.abs(expected - actual) should be < 0.05 }
  }

}
