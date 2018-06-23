/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
