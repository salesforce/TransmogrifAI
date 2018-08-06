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

package com.salesforce.op.stages

import com.salesforce.op.features._
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.RichDataType._
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.ml.{PipelineStage, Transformer}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

import scala.reflect.runtime.universe.TypeTag
import scala.util.{Success, Try}


/**
 * Optimus Prime Base Pipeline Stage allowing to specify arbitrary Input and Output Feature types
 *
 * Internally the stage operates and serializes the TransientFeature. However to maintain usability
 * for users, the getters will return FeatureLike objects. It is important that during development
 * these features are not captured into a UDF since the entire DAG will be serialized onto worker
 * nodes. All stage method, when referring to inputs, should access them via HasIn[1,2,3,4,N] traits.
 */
trait OpPipelineStageBase extends OpPipelineStageParams with MLWritable {
  self: PipelineStage =>

  type OutputFeatures

  /**
   * Short unique name of the operation this stage performs
   *
   * @return operation name
   */
  def operationName: String

  /**
   * Stage unique name consisting of the stage operation name and uid
   *
   * @return stage name
   */
  final def stageName: String = s"${operationName}_$uid"

  /**
   * Input features that will be used by the stage
   *
   * @return feature of type InputFeatures
   */
  final def setInput(features: InputFeatures): this.type = {
    setInputFeatures(features)
    onSetInput()
    this
  }

  /**
   * Function to be called on setInput
   */
  protected def onSetInput(): Unit = {}

  /**
   * Output features that will be created by this stage
   *
   * @return feature of type OutputFeatures
   */
  def getOutput(): OutputFeatures

  /**
   * Function to convert OutputFeatures to an Array of FeatureLike
   *
   * @return an Array of FeatureLike
   */
  protected implicit def outputAsArray(out: OutputFeatures): Array[OPFeature]

  /**
   * Check if the stage is serializable
   *
   * @return Failure if not serializable
   */
  def checkSerializable: Try[Unit] = Success(Unit)

  /**
   * This function translates the input and output features into spark schema checks and changes that will occur on
   * the underlying data frame
   *
   * @param schema schema of the input data frame
   * @return a new schema with the output features added
   */
  final override def transformSchema(schema: StructType): StructType = {
    val schemaFields = schema.fieldNames.toSet

    getInputFeatures().foreach { f =>
      require(schemaFields.contains(f.name),
        s"Input feature '${f.name}' does exist in the schema:\n${schema.treeString}"
      )
      val inputType = schema(f.name).dataType
      val expectedInputType = FeatureSparkTypes.sparkTypeOf(f.wtt)
      if (!inputType.equalsIgnoreNullability(expectedInputType)) {
        throw new IllegalArgumentException(
          s"Input data type $inputType of '${f.name}' does not match the expected type $expectedInputType"
        )
      }
    }

    val outputs: Array[OPFeature] = getOutput()
    outputs.foreach { out =>
      if (schemaFields.contains(out.name)) {
        val sid = Option(out.originStage).map(_.uid).orNull
        throw new IllegalArgumentException(
          s"Output column ${out.name} produced by stage $sid already exists"
        )
      }
    }

    val metadata = getMetadata()
    val outputFields = schema.fields ++ outputs.map(FeatureSparkTypes.toStructField(_, metadata))
    StructType(outputFields)
  }

  /**
   * This method is used to make a copy of the instance with new parameters in several methods in spark internals
   * Default will find the constructor and make a copy for any class
   * (AS LONG AS ALL CONSTRUCTOR PARAMS ARE VALS, this is why type tags are written as implicit vals in base classes).
   *
   * Note: that the convention in spark is to have the uid be a constructor argument,
   * so that copies will share a uid with the original (developers should follow this convention).
   *
   * @param extra new parameters want to add to instance
   * @return a new instance with the same uid
   */
  final override def copy(extra: ParamMap): this.type = {
    val copy = ReflectionUtils.copy(this).asInstanceOf[this.type]
    copyValues(copy, extra)
  }

  final override def write: MLWriter = new OpPipelineStageWriter(this)

}


/**
 * Optimus Prime Base Pipeline Stage allowing to specify Input Feature types and a single Output Feature type
 *
 * @tparam O output feature type
 */
trait OpPipelineStage[O <: FeatureType] extends OpPipelineStageBase {
  self: PipelineStage =>

  type InputFeatures
  final override type OutputFeatures = FeatureLike[O]

  final override def outputAsArray(out: OutputFeatures): Array[OPFeature] = Array[OPFeature](out)

  protected[op] def outputFeatureUid: String

  /**
   * The name that the output feature and column should have if unset will use the default name
   */
  final private[op] val outputFeatureName = new Param[String](
    parent = this, name = "outputFeatureName",
    doc = "output name that overrides default output name for feature made by this stage"
  )

  def setOutputFeatureName(name: String): this.type = set(outputFeatureName, name)

  /**
   * Name of output feature (i.e. column created by this stage)
   */
  final def getOutputFeatureName: String =
    get(outputFeatureName).getOrElse(makeOutputName(outputFeatureUid, getTransientFeatures()))

  /**
   * Should output feature be a response? Yes, if any of the input features are.
   * @return true if the the output feature should be a response
   */
  def outputIsResponse: Boolean = getTransientFeatures().exists(_.isResponse)

}

private[op] trait AllowLabelAsInput[O <: FeatureType] extends OpPipelineStage[O] {
  self: PipelineStage =>

  // Since we use label information we override the response function allowing label leakage here,
  // except the cases where both input features are responses
  abstract override def outputIsResponse: Boolean = getTransientFeatures().forall(_.isResponse)

}

/**
 * Pipeline stage of Feature type I to O
 *
 * @tparam I input feature type
 * @tparam O output feature type
 */
trait OpPipelineStage1[I <: FeatureType, O <: FeatureType] extends OpPipelineStage[O] with HasIn1 {
  self: PipelineStage =>

  implicit val tto: TypeTag[O]
  implicit val ttov: TypeTag[O#Value]

  final override type InputFeatures = FeatureLike[I]

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 1

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = Array(in)

  protected[op] override def outputFeatureUid: String = FeatureUID[O](uid)

  override def getOutput(): FeatureLike[O] = new Feature[O](
    uid = outputFeatureUid,
    name = getOutputFeatureName,
    originStage = this,
    isResponse = outputIsResponse,
    parents = getInputFeatures()
  )(tto)

}


/**
 * Pipeline stage of Feature type I to Features O1 and O2
 *
 * @tparam I  input feature type
 * @tparam O1 first output feature type
 * @tparam O2 second output feature type
 */
trait OpPipelineStage1to2[I <: FeatureType, O1 <: FeatureType, O2 <: FeatureType]
  extends OpPipelineStageBase with HasIn1 {
  self: PipelineStage =>

  final override type InputFeatures = FeatureLike[I]
  final override type OutputFeatures = (FeatureLike[O1], FeatureLike[O2])

  def stage1OperationName: String

  def stage2OperationName: String

  def operationName: String = s"${stage1OperationName}_${stage2OperationName}"

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 1

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = Array(in)

  final override def outputAsArray(out: OutputFeatures): Array[OPFeature] = {
    out.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

}

/**
 * Pipeline stage of Feature type I to Features O1, O2 and O3
 *
 * @tparam I  input feature type
 * @tparam O1 first output feature type
 * @tparam O2 second output feature type
 * @tparam O3 third output feature type
 */
trait OpPipelineStage1to3[I <: FeatureType, O1 <: FeatureType, O2 <: FeatureType, O3 <: FeatureType]
  extends OpPipelineStageBase with HasIn1 {
  self: PipelineStage =>

  final override type InputFeatures = FeatureLike[I]
  final override type OutputFeatures = (FeatureLike[O1], FeatureLike[O2], FeatureLike[O3])

  def stage1OperationName: String

  def stage2OperationName: String

  def stage3OperationName: String

  def operationName: String = s"${stage1OperationName}_${stage2OperationName}_${stage3OperationName}"

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 1

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = Array(in)

  final override def outputAsArray(out: OutputFeatures): Array[OPFeature] = {
    out.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

}


/**
 * Pipeline stage of Feature type I1 and I2 to Feature of type O
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam O  output feature type
 */
trait OpPipelineStage2[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
  extends OpPipelineStage[O] with HasIn1 with HasIn2 {
  self: PipelineStage =>

  implicit val tto: TypeTag[O]
  implicit val ttov: TypeTag[O#Value]

  final override type InputFeatures = (FeatureLike[I1], FeatureLike[I2])

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 2

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = {
    in.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

  protected[op] override def outputFeatureUid: String = FeatureUID[O](uid)

  override def getOutput(): FeatureLike[O] = new Feature[O](
    uid = outputFeatureUid,
    name = getOutputFeatureName,
    originStage = this,
    isResponse = outputIsResponse,
    parents = getInputFeatures()
  )(tto)

}

/**
 * Pipeline stage of Feature type I1 and I2 to Feature of type O1 and O2
 *
 * @tparam I1 input feature type 1
 * @tparam I2 input feature type 2
 * @tparam O1 first output feature type
 * @tparam O2 second output feature type
 */
trait OpPipelineStage2to2[I1 <: FeatureType, I2 <: FeatureType, O1 <: FeatureType, O2 <: FeatureType]
  extends OpPipelineStageBase with HasIn1 with HasIn2 {
  self: PipelineStage =>

  final override type InputFeatures = (FeatureLike[I1], FeatureLike[I2])
  final override type OutputFeatures = (FeatureLike[O1], FeatureLike[O2])

  def stage1OperationName: String

  def stage2OperationName: String

  def operationName: String = s"${stage1OperationName}_${stage2OperationName}"

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 2

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = {
    in.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

  final override def outputAsArray(out: OutputFeatures): Array[OPFeature] = {
    out.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }
}

/**
 * Pipeline stage of Feature type I1 and I2 to Features O1, O2 and O3
 *
 * @tparam I1 input feature type 1
 * @tparam I2 input feature type 2
 * @tparam O1 first output feature type
 * @tparam O2 second output feature type
 * @tparam O3 third output feature type
 */
trait OpPipelineStage2to3[I1 <: FeatureType, I2 <: FeatureType, O1 <: FeatureType, O2 <: FeatureType, O3 <: FeatureType]
  extends OpPipelineStageBase with HasIn1 with HasIn2 {
  self: PipelineStage =>

  final override type InputFeatures = (FeatureLike[I1], FeatureLike[I2])
  final override type OutputFeatures = (FeatureLike[O1], FeatureLike[O2], FeatureLike[O3])

  def stage1OperationName: String

  def stage2OperationName: String

  def stage3OperationName: String

  def operationName: String = s"${stage1OperationName}_${stage2OperationName}_${stage3OperationName}"

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 2

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = {
    in.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

  final override def outputAsArray(out: OutputFeatures): Array[OPFeature] = {
    out.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }
}

/**
 * Pipeline stage of Feature type I1, I2 and I3 to Feature of type O
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam O  output feature type
 */
trait OpPipelineStage3[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, O <: FeatureType]
  extends OpPipelineStage[O] with HasIn1 with HasIn2 with HasIn3 {
  self: PipelineStage =>

  implicit val tto: TypeTag[O]
  implicit val ttov: TypeTag[O#Value]

  final override type InputFeatures = (FeatureLike[I1], FeatureLike[I2], FeatureLike[I3])

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 3

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = {
    in.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

  protected[op] override def outputFeatureUid: String = FeatureUID[O](uid)

  override def getOutput(): FeatureLike[O] = new Feature[O](
    uid = outputFeatureUid,
    name = getOutputFeatureName,
    originStage = this,
    isResponse = outputIsResponse,
    parents = getInputFeatures()
  )(tto)
}

/**
 * Pipeline stage of Feature type I1, I2 and I3 to Feature of type O1 and O2
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam O1 first output feature type
 * @tparam O2 second output feature type
 */
trait OpPipelineStage3to2[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, O1 <: FeatureType, O2 <: FeatureType]
  extends OpPipelineStageBase with HasIn1 with HasIn2 with HasIn3 {
  self: PipelineStage =>

  final override type InputFeatures = (FeatureLike[I1], FeatureLike[I2], FeatureLike[I3])
  final override type OutputFeatures = (FeatureLike[O1], FeatureLike[O2])

  def stage1OperationName: String

  def stage2OperationName: String

  def operationName: String = s"${stage1OperationName}_${stage2OperationName}"

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 3

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = {
    in.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

  final override def outputAsArray(out: OutputFeatures): Array[OPFeature] = {
    out.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }
}


/**
 * Pipeline stage of Feature type I1, I2, I3, and I4 to Feature of type O
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  output feature type
 */
trait OpPipelineStage4[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType, O <: FeatureType]
  extends OpPipelineStage[O] with HasIn1 with HasIn2 with HasIn3 with HasIn4 {
  self: PipelineStage =>

  implicit val tto: TypeTag[O]
  implicit val ttov: TypeTag[O#Value]

  final override type InputFeatures = (FeatureLike[I1], FeatureLike[I2], FeatureLike[I3], FeatureLike[I4])

  final override def checkInputLength(features: Array[_]): Boolean = features.length == 4

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = {
    in.productIterator.map(_.asInstanceOf[OPFeature]).toArray
  }

  protected[op] override def outputFeatureUid: String = FeatureUID[O](uid)

  override def getOutput(): FeatureLike[O] = new Feature[O](
    uid = outputFeatureUid,
    name = getOutputFeatureName,
    originStage = this,
    isResponse = outputIsResponse,
    parents = getInputFeatures()
  )(tto)
}


/**
 * Pipeline stage of multiple Features of type I to Feature of type O
 *
 * @tparam I input feature type
 * @tparam O output feature type
 */
trait OpPipelineStageN[I <: FeatureType, O <: FeatureType] extends OpPipelineStage[O] with HasInN {
  self: PipelineStage =>

  implicit val tto: TypeTag[O]
  implicit val ttov: TypeTag[O#Value]

  final override type InputFeatures = Array[FeatureLike[I]]

  final override def checkInputLength(features: Array[_]): Boolean = features.length > 0

  final override def inputAsArray(in: InputFeatures): Array[OPFeature] = {
    in.asInstanceOf[Array[OPFeature]]
  }

  final def setInput(features: FeatureLike[I]*): this.type = super.setInput(features.toArray)

  protected[op] override def outputFeatureUid: String = FeatureUID[O](uid)

  override def getOutput(): FeatureLike[O] = new Feature[O](
    uid = outputFeatureUid,
    name = getOutputFeatureName,
    originStage = this,
    isResponse = outputIsResponse,
    parents = getInputFeatures()
  )(tto)
}

/**
 * Trait to mix into transformers that indicates their transform functions can be combined into a single stage
 */
private[op] trait OpTransformer {
  self: OpPipelineStage[_] with Transformer =>

  /**
   * Feature name (key) -> value lookup, e.g Row, Map etc.
   */
  type KeyValue = String => Any

  /**
   * Creates a transform function to transform Row to a value
   * @return a transform function to transform Row to a value
   */
  def transformRow: Row => Any = r => transformKeyValue(r.getAny)

  /**
   * Creates a transform function to transform Map to a value
   * @return a transform function to transform Map to a value
   */
  def transformMap: Map[String, Any] => Any = m => transformKeyValue(m.apply)

  /**
   * Creates a transform function to transform any key/value to a value
   * @return a transform function to transform any key/value to a value
   */
  def transformKeyValue: KeyValue => Any
}
