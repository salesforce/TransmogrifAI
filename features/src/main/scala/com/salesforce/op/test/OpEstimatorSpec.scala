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

package com.salesforce.op.test

import java.io.File

import com.salesforce.op.features.types._
import com.salesforce.op.stages._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset
import org.scalactic.Equality
import org.scalatest.events.{Event, TestFailed}
import org.scalatest.{Args, Reporter}

import scala.collection.mutable.ArrayBuffer
import scala.reflect._
import scala.reflect.runtime.universe._

/**
 * Base test class for testing OP estimator instances.
 * Includes common tests for fitting estimator and verifying the fitted model.
 *
 * @tparam O             output feature type
 * @tparam ModelType     model type produced by this estimator
 * @tparam EstimatorType type of the estimator being tested
 */
abstract class OpEstimatorSpec[O <: FeatureType : WeakTypeTag : ClassTag,
ModelType <: Model[ModelType] with OpPipelineStage[O] with OpTransformer : ClassTag,
EstimatorType <: Estimator[ModelType] with OpPipelineStage[O] : ClassTag]
  extends OpPipelineStageSpec[O, EstimatorType] {

  /**
   * Input Dataset to fit & transform
   */
  val inputData: Dataset[_]

  /**
   * Estimator instance to be tested
   */
  val estimator: EstimatorType

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult: Seq[O]

  final override lazy val stage = estimator

  /**
   * Model (transformer) to fit
   */
  final lazy val model: ModelType = estimator.fit(inputData)

  it should "fit a model" in {
    model should not be null
    model shouldBe a[ModelType]
  }

  it should behave like modelSpec()

  it should "have fitted a model that matches the estimator" in {
    withClue("Model doesn't have a parent:") {
      model.hasParent shouldBe true
    }
    withClue("Model parent should be the original estimator instance:") {
      model.parent shouldBe estimator
    }
    withClue("Model and estimator output feature names don't match:") {
      model.getOutputFeatureName shouldBe estimator.getOutputFeatureName
    }
    assert(model.asInstanceOf[OpPipelineStageBase], estimator, expectSameClass = false)
  }

  // TODO: test metadata


  /**
   * Register all model spec tests
   */
  private def modelSpec(): Unit = {
    // Define transformer spec for the fitted model reusing the same inputs & Spark context
    val modelSpec = new OpTransformerSpec[O, ModelType] {
      override implicit val featureTypeEquality: Equality[O] = OpEstimatorSpec.this.featureTypeEquality
      override implicit val seqEquality: Equality[Seq[O]] = OpEstimatorSpec.this.seqEquality
      lazy val transformer: ModelType = model.setInputFeatureArray(estimator.getInputFeatures())
      lazy val inputData: Dataset[_] = OpEstimatorSpec.this.inputData
      lazy val expectedResult: Seq[O] = OpEstimatorSpec.this.expectedResult
      override implicit lazy val spark = OpEstimatorSpec.this.spark
      override def specName: String = "model"
      override def tempDir: File = OpEstimatorSpec.this.tempDir
    }

    // Register all model spec tests
    for {
      testName <- modelSpec.testNames
    } registerTest(testName) {
      // Run test & collect failures
      val failures = ArrayBuffer.empty[TestFailed]
      val reporter = new Reporter {
        def apply(event: Event): Unit = event match {
          case f: TestFailed => failures += f
          case _ =>
        }
      }
      // Note: We set 'runTestInNewInstance = true' to avoid restarting Spark context on every test run
      val args = Args(reporter, runTestInNewInstance = true)
      modelSpec.run(testName = Some(testName), args = args)

      // Propagate the failure if any
      for {failure <- failures.headOption} {
        failure.throwable.map(fail(failure.message, _)).getOrElse(fail(failure.message))
      }
    }
  }

}
