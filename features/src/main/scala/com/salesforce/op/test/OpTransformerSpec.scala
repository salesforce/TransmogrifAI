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

import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureLike, FeatureSparkTypes}
import com.salesforce.op.stages._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.encoders.RowEncoder

import scala.reflect._
import scala.reflect.runtime.universe._

/**
 * Base test class for testing [[OpPipelineStage]] instances (transformers or estimators).
 * Includes common tests for schema and data transformations.
 *
 * @tparam O               output feature type
 * @tparam TransformerType type of the transformer being tested
 */
abstract class OpTransformerSpec[O <: FeatureType,
TransformerType <: OpPipelineStage[O] with Transformer with OpTransformer]
(
  implicit val cto: ClassTag[O], val wto: WeakTypeTag[O], val ttc: ClassTag[TransformerType]
) extends OpPipelineStageSpec[O, TransformerType] with TransformerSpecCommon[O, TransformerType] {

  it should "transform rows" in {
    val rows = inputData.toDF().collect()
    val res: Seq[O] = rows.view.map(row => transformer.transformRow(row)).map(convert.fromSpark)
    res shouldEqual expectedResult
  }
  it should "transform maps" in {
    val rows = inputData.toDF().collect()
    val inputNames = transformer.getTransientFeatures().map(_.name)
    val maps = rows.view.map(row => inputNames.map(name => name -> row.getAny(name)).toMap)
    val res: Seq[O] = maps.map(transformer.transformMap).map(convert.fromSpark)
    res shouldEqual expectedResult
  }
  it should "transform key/value" in {
    val rows = inputData.toDF().collect()
    val res: Seq[O] = rows.view.map(row => transformer.transformKeyValue(row.getAny)).map(convert.fromSpark)
    res shouldEqual expectedResult
  }
  it should "transform data after being loaded" in {
    val loaded = writeAndRead(stage)
    val transformed = loaded.asInstanceOf[TransformerType].transform(inputData)
    val output = loaded.getOutput().asInstanceOf[FeatureLike[O]]
    val res: Seq[O] = transformed.collect(output)(convert, classTag[O]).toSeq
    res shouldEqual expectedResult
  }
  it should "transform data after being loaded without spark" in {
    val loaded = writeAndRead(stage, asSpark = false)
    val transformed = loaded.asInstanceOf[TransformerType].transform(inputData)
    val output = loaded.getOutput().asInstanceOf[FeatureLike[O]]
    val res: Seq[O] = transformed.collect(output)(convert, classTag[O]).toSeq
    res shouldEqual expectedResult
  }
}

/**
 * Common test transformer functionality for [[OpTransformerSpec]] and [[SwTransformerSpec]] specs
 *
 * @tparam O               output feature type
 * @tparam TransformerType type of the transformer being tested
 */
private[test] trait TransformerSpecCommon[O <: FeatureType, TransformerType <: OpPipelineStage[O] with Transformer] {
  self: OpPipelineStageSpec[O, TransformerType] =>

  /**
   * Transformer instance to be tested
   */
  val transformer: TransformerType

  /**
   * Input Dataset to transform
   */
  val inputData: Dataset[_]

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult: Seq[O]

  implicit def cto: ClassTag[O]
  implicit def wto: WeakTypeTag[O]
  protected lazy val convert: FeatureTypeSparkConverter[O] = FeatureTypeSparkConverter[O]()

  final override lazy val stage = transformer

  it should "be json writable/readable" in {
    transformer.transform(inputData) // needs to have been called so the params on wrapped stages are set to save
    val loaded = writeAndRead(transformer)
    assert(loaded, transformer)


  }
  it should "be json writable/readable as a local stage" in {
    transformer.transform(inputData) // needs to have been called so the params on wrapped stages are set to save
    val loaded = writeAndRead(transformer, asSpark = false)
    loaded match {
      case l if l.isInstanceOf[SparkWrapperParams[_]] =>
        val sparkWrapped = l.asInstanceOf[SparkWrapperParams[_]]
        sparkWrapped.getLocalMlStage().isDefined shouldBe true
        sparkWrapped.getSparkMlStage().isEmpty shouldBe true
      case _ =>
    }
    assert(loaded, transformer)
  }
  it should "transform schema" in {
    val transformedSchema = transformer.transformSchema(inputData.schema)
    val output = transformer.getOutput()
    val validationResults =
      FeatureSparkTypes.validateSchema(transformedSchema, transformer.getInputFeatures() :+ output)
    if (validationResults.nonEmpty) {
      fail("Dataset schema is invalid. Errors: " + validationResults.mkString("'", "','", "'"))
    }
  }
  it should "transform data" in {
    val transformed = transformer.transform(inputData)
    val output = transformer.getOutput()
    val res: Seq[O] = transformed.collect(output)(convert, classTag[O]).toSeq
    res shouldEqual expectedResult
  }
  it should "transform empty data" in {
    val empty = spark.emptyDataset(RowEncoder(inputData.schema))
    val transformed = transformer.transform(empty)
    val output = transformer.getOutput()
    val res: Seq[O] = transformed.collect(output)(convert, classTag[O]).toSeq
    res.size shouldBe 0
  }

  /**
   * A helper function to write and read stage into savePath
   *
   * @param stage    stage instance to write and then read
   * @param savePath Spark stage save path
   * @return read stage
   */
  protected def writeAndRead(
    stage: TransformerType, savePath: String = stageSavePath, asSpark: Boolean = true
  ): OpPipelineStageBase = {
    val json = new OpPipelineStageWriter(stage).overwrite().writeToJsonString(savePath)
    val features = stage.getInputFeatures().flatMap(_.allFeatures)
    new OpPipelineStageReader(features).loadFromJsonString(json, savePath, asSpark)
  }

  /**
   * Spark stage save path
   */
  protected def stageSavePath: String = s"$tempDir/${specName.filter(_.isLetterOrDigit)}-${System.currentTimeMillis()}"

}
