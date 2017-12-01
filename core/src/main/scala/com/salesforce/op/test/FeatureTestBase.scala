/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.utils.spark.RichDataset.RichDataset
import org.apache.spark.ml.{Estimator, Transformer}
import org.apache.spark.sql.Dataset
import org.scalatest.prop.PropertyChecks
import org.scalatest.{Assertion, Suite}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/**
 * Common functionality for testing Feature operations (transformers / estimators)
 */
trait FeatureTestBase extends TestSparkContext with PropertyChecks {
  self: Suite =>

  /**
   * Test unary Feature operation (transformer / estimator)
   *
   * @param op unary Feature operation
   * @tparam A input feature type
   * @tparam C output feature type
   * @return feature tester
   */
  def testOp[A <: FeatureType : TypeTag,
  C <: FeatureType : TypeTag : FeatureTypeSparkConverter : ClassTag]
  (
    op: FeatureLike[A] => FeatureLike[C]
  ): UnaryTester[A, C] = new UnaryTester[A, C] {
    def of(v: A*): Checker[C] = new Checker[C] {
      def expecting(z: C*): Assertion = {
        val (data, f1) = TestFeatureBuilder[A](v)
        val f = op(f1)
        checkFeature(f, data, expected = z, clue = s"Testing ${f.originStage.operationName} on $v: ")
      }
    }
  }

  /**
   * Test binary Feature operation (transformer / estimator)
   *
   * @param op binary Feature operation
   * @tparam A first input feature type
   * @tparam B second input feature type
   * @tparam C output feature type
   * @return feature tester
   */
  def testOp[A <: FeatureType : TypeTag,
  B <: FeatureType : TypeTag,
  C <: FeatureType : TypeTag : FeatureTypeSparkConverter : ClassTag]
  (
    op: FeatureLike[A] => FeatureLike[B] => FeatureLike[C]
  ): BinaryTester[A, B, C] = new BinaryTester[A, B, C] {
    def of(v: (A, B)*): Checker[C] = new Checker[C] {
      def expecting(z: C*): Assertion = {
        val (data, f1, f2) = TestFeatureBuilder[A, B](v)
        val f = op(f1)(f2)
        checkFeature(f, data, expected = z, clue = s"Testing ${f.originStage.operationName} on $v: ")
      }
    }
  }

  sealed abstract class UnaryTester[A <: FeatureType,
  C <: FeatureType : TypeTag : FeatureTypeSparkConverter : ClassTag] {
    def of(x: A*): Checker[C]
  }

  sealed abstract class BinaryTester[A <: FeatureType,
  B <: FeatureType, C <: FeatureType : TypeTag : FeatureTypeSparkConverter : ClassTag] {
    def of(x: A, y: B): Checker[C] = of((x, y))
    def of(x: (A, B)*): Checker[C]
  }

  sealed abstract class Checker[C <: FeatureType : TypeTag : FeatureTypeSparkConverter : ClassTag] {
    def expecting(z: C*): Assertion

    protected def checkFeature(f: FeatureLike[C], data: Dataset[_], clue: String, expected: Seq[C]): Assertion = {
      val transformed = f.originStage match {
        case e: Estimator[_] => e.fit(data).transform(data)
        case t: Transformer => t.transform(data)
      }
      withClue(clue)(
        new RichDataset(transformed).collect[C](f) should contain theSameElementsInOrderAs expected
      )
    }
  }

}
