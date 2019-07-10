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

package com.salesforce.op.tensorflow

import java.nio.IntBuffer
import java.nio.file.Files
import java.util.zip.{ZipEntry, ZipInputStream}

import com.google.common.io.Resources
import com.robrua.nlp.bert.FullTokenizer
import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.stages.{OpPipelineStageReaderWriter, ReaderWriter}
import org.apache.commons.io.{FileUtils, IOUtils}
import org.apache.spark.ml.linalg.Vectors
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.tensorflow._
import org.bytedeco.tensorflow.global.tensorflow._
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.io.Source
import scala.util.Try


@ReaderWriter(classOf[BERTModelVectorizerReaderWriter])
class BERTModelVectorizer
(
  val modelLoader: BERTModelLoader,
  uid: String = UID[BERTModelVectorizer]
) extends UnaryTransformer[Text, OPVector](uid = uid, operationName = "bert") with AutoCloseable {

  @transient private lazy val bertModel: BERTModel = modelLoader.model

  def transformFn: Text => OPVector = {
    case SomeValue(Some(s)) => Vectors.dense(bertModel(s).map(_.toDouble)).toOPVector
    case _ => OPVector.empty // TODO: is it ok to return an empty vector here?
  }

  def close(): Unit = if (bertModel != null) bertModel.close()
}

class BERTModelVectorizerReaderWriter extends OpPipelineStageReaderWriter[BERTModelVectorizer] {

  def read(stageClass: Class[BERTModelVectorizer], json: JValue): Try[BERTModelVectorizer] = Try {
    val modelLoaderJson = (json \ "modelLoader").extract[JObject]
    val modelLoader = (modelLoaderJson \ "className").extract[String] match {
      case c if c == classOf[BERTModelResourceLoader].getName =>
        val resource = (modelLoaderJson \ "resource").extract[String]
        new BERTModelResourceLoader(resource)
      case c =>
        throw new RuntimeException(s"Unknown BERT model loader class: $c")
    }
    new BERTModelVectorizer(
      uid = (json \ "uid").extract[String],
      modelLoader = modelLoader
    )
  }

  def write(stage: BERTModelVectorizer): Try[JValue] = Try {
    val modelLoader: JValue = stage.modelLoader match {
      case r: BERTModelResourceLoader => ("className" -> r.getClass.getName) ~ ("resource" -> r.resource)
      case r => "className" -> r.getClass.getName
    }
    ("uid" -> stage.uid) ~ ("modelLoader" -> modelLoader)
  }
}

case class BERTModelConfig
(
  doLowerCase: Boolean,
  inputIds: String,
  inputMask: String,
  segmentIds: String,
  pooledOutput: String,
  sequenceOutput: String,
  maxSequenceLength: Int
)

case class BERTModel
(
  config: BERTModelConfig,
  modelBundle: SavedModelBundle,
  tokenizer: FullTokenizer
) extends AutoCloseable {
  private val startTokenId = tokenizer.convert(Array("[CLS]"))(0)
  private val separatorTokenId = tokenizer.convert(Array("[SEP]"))(0)

  /**
   * BERT model inputs
   *
   * @param inputIds   inputIds are the indexes in the vocabulary for each token in the sequence
   * @param inputMask  is a binary mask that shows which inputIds have valid data in them
   * @param segmentIds are meant to distinguish paired sequences during training tasks
   */
  case class Inputs(inputIds: Tensor, inputMask: Tensor, segmentIds: Tensor) extends AutoCloseable {
    def close(): Unit = {
      inputIds.close()
      inputMask.close()
      segmentIds.close()
    }
  }

  def apply(s: String): Array[Float] = {
    val tokens = tokenizer.tokenize(s)
    val ids = tokenizer.convert(tokens)
    val inputs = prepareInputs(tokens, ids, config.maxSequenceLength)
    val input_feed = new StringTensorPairVector(
      Array(config.inputIds, config.inputMask, config.segmentIds),
      Array(inputs.inputIds, inputs.inputMask, inputs.segmentIds)
    )
    val outputs = new TensorVector
    modelBundle.session()
      .Run(input_feed, new StringVector(config.pooledOutput), new StringVector, outputs)
      .errorIfNotOK()

    outputs.get(0).asFloatArray
  }

  /**
   * In BERT:
   * inputIds are the indexes in the vocabulary for each token in the sequence
   * inputMask is a binary mask that shows which inputIds have valid data in them
   * segmentIds are meant to distinguish paired sequences during training tasks.
   * Here they're always 0 since we're only doing inference.
   */
  private def prepareInputs(tokens: Array[String], ids: Array[Int], maxSequenceLength: Int): Inputs = {
    val inputIdsT = new Tensor(DT_INT32, new TensorShape(1, maxSequenceLength))
    val inputMaskT = new Tensor(DT_INT32, new TensorShape(1, maxSequenceLength))
    val segmentIdsT = new Tensor(DT_INT32, new TensorShape(1, maxSequenceLength))

    val inputIds = inputIdsT.createBuffer[IntBuffer]()
    val inputMask = inputMaskT.createBuffer[IntBuffer]()
    val segmentIds = segmentIdsT.createBuffer[IntBuffer]()

    inputIds.put(startTokenId)
    inputMask.put(1)
    segmentIds.put(0)

    var i = 0
    while (i < ids.length && i < maxSequenceLength - 2) {
      inputIds.put(ids(i))
      inputMask.put(1)
      segmentIds.put(0)
      i += 1
    }
    inputIds.put(separatorTokenId)
    inputMask.put(1)
    segmentIds.put(0)

    while(inputIds.position() < maxSequenceLength) {
      inputIds.put(0)
      inputMask.put(0)
      segmentIds.put(0)
    }
    inputIds.rewind()
    inputMask.rewind()
    segmentIds.rewind()

    Inputs(inputIdsT, inputMaskT, segmentIdsT)
  }

  def close(): Unit = if (modelBundle.session() != null) modelBundle.session().close()
}

trait BERTModelLoader extends Serializable {
  def model: BERTModel
}

class BERTModelResourceLoader(val resource: String) extends BERTModelLoader {

  lazy val model: BERTModel = {
    val res = Resources.getResource(resource)
    val bertModelDir = Files.createTempDirectory("bert-model-" + System.currentTimeMillis())

    try {
      val bertZip = new ZipInputStream(Resources.asByteSource(res).openBufferedStream())
      var entry: ZipEntry = bertZip.getNextEntry
      while (entry != null) {
        val path = bertModelDir.resolve(entry.getName)
        if (entry.getName.endsWith("/")) Files.createDirectories(path)
        else {
          Files.createFile(path)
          val output = Files.newOutputStream(path)
          IOUtils.copy(bertZip, output)
        }
        bertZip.closeEntry()
        entry = bertZip.getNextEntry
      }

      // Load model assets
      implicit val formats = DefaultFormats
      val assets = bertModelDir.resolve("assets")
      val config = Source.fromFile(assets.resolve("model.json").toFile).getLines.mkString
      val modelConfig = parse(config).extract[BERTModelConfig]

      // Prepare the tokenizer
      val tokenizer = new FullTokenizer(assets.resolve("vocab.txt").toFile, modelConfig.doLowerCase)

      // Load the saved model itself
      val tags = new StringUnorderedSet()
      tags.insert(new BytePointer("serve"))
      val modelBundle = new SavedModelBundle()
      LoadSavedModel(
        new SessionOptions(), new RunOptions(), bertModelDir.toAbsolutePath.toFile.toString, tags, modelBundle
      )

      BERTModel(config = modelConfig, modelBundle = modelBundle, tokenizer = tokenizer)
    } finally {
      FileUtils.deleteDirectory(bertModelDir.toFile)
    }
  }

}
