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

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.stages._
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.ParamMap
import org.scalatest._

import scala.reflect._
import scala.reflect.runtime.universe._
import scala.util.Failure


/**
 * Spec for testing [[OpPipelineStage]] instances (transformers or estimators).
 * Includes common tests for output feature, copy, serialization, json read/write etc.
 *
 * @tparam O         output feature type
 * @tparam StageType [[OpPipelineStage]] type being tested (transformer or estimator)
 */
abstract class OpPipelineStageSpec[O <: FeatureType : WeakTypeTag : ClassTag,
StageType <: OpPipelineStage[O] : ClassTag]
  extends FlatSpec
    with FeatureTypeEquality[O]
    with TestSparkContext
    with OpPipelineStageAsserts {

  /**
   * [[OpPipelineStage]] instance to be tested
   */
  val stage: StageType

  /**
   * Spec name (StageType[O] by default)
   */
  def specName: String = Spec[O, StageType]

  specName should "produce output feature" in {
    val output = stage.getOutput()
    output shouldBe new Feature[O](
      name = stage.getOutputFeatureName,
      originStage = stage,
      isResponse = stage.outputIsResponse,
      parents = stage.getInputFeatures()
    )
  }
  it should "copy" in {
    val copy = stage.copy(new ParamMap())
    copy shouldBe a[StageType]
    assert(copy, stage)
  }
  it should "be serializable" in {
    stage.checkSerializable match {
      case Failure(e) => fail("Stage is not serializable", e)
      case _ =>
    }
  }

}


/**
 * Stage assertion for [[OpPipelineStage]]
 */
trait OpPipelineStageAsserts extends AppendedClues {
  self: Matchers =>

  /**
   * Assert stage instances
   *
   * @param stage           instance to assert
   * @param expected        instance to assert against
   * @param expectSameClass should expect the same class or not
   * @return
   */
  def assert(stage: OpPipelineStageBase, expected: OpPipelineStageBase, expectSameClass: Boolean = true): Assertion = {
    def stageType(s: OpPipelineStageBase) = if (s.isInstanceOf[Estimator[_]]) "estimator" else "transformer"
    lazy val stageClue =
      if (expectSameClass) s", while asserting ${stage.getClass.getSimpleName} ${stageType(stage)}."
      else {
        s", while asserting ${stage.getClass.getSimpleName} ${stageType(stage)} " +
          s"against ${expected.getClass.getSimpleName} ${stageType(expected)}."
      }
    def clue[T](msg: String)(fun: => T) = { withClue(msg)(fun) } withClue stageClue

    if (expectSameClass) {
      clue("Stage classes don't match:") {
        stage.getClass shouldBe expected.getClass
      }
      clue("Params are not the same:") {
        stage.params should contain theSameElementsAs expected.params
      }
      expected.params.foreach { p =>
        clue(s"Param '${p.name}' should exist:") {
          stage.hasParam(p.name) shouldBe expected.hasParam(p.name)
        }
        // TODO: add params value comparison (note: can be tricky)
        // withClue(s"Param '${p.name}' values do not match:") {
        //   stage.get(p) shouldBe expected.get(p)
        // }
      }
    }
    clue("Stage UIDs don't match:") {
      stage.uid shouldBe expected.uid
    }
    clue("Stage outputs don't match:") {
      stage.getOutput() shouldBe expected.getOutput()
    }
    clue("Operation names don't match:") {
      stage.operationName shouldBe expected.operationName
    }
    clue("Stage names don't match:") {
      stage.stageName shouldBe expected.stageName
    }
    clue("Transient features don't match:") {
      stage.getTransientFeatures() should contain theSameElementsAs expected.getTransientFeatures()
    }
    clue("Input features don't match:") {
      stage.getInputFeatures() should contain theSameElementsAs expected.getInputFeatures()
    }
    clue("Input schemas don't match:") {
      stage.getInputSchema().fields.size shouldEqual expected.getInputSchema().fields.size
      stage.getInputSchema().fields.zip(expected.getInputSchema().fields).foreach{
        case (sf, ef) =>
          sf.name shouldBe ef.name
          sf.dataType shouldBe ef.dataType
          // Should not rely on InputSchema anymore to pass around metadata
          // sf.metadata.deepEquals(ef.metadata) shouldBe true
          sf.nullable shouldBe ef.nullable
      }
    }
    clue("Metadata values don't match:") {
      stage.getMetadata().deepEquals(expected.getMetadata()) shouldBe true
    }
  }
}
