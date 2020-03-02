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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.OpPipelineStageBase
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.spark.RichVector._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.feature.HashingTF

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe.TypeTag

/**
 * Generic hashing vectorizer to convert features of type OPCollection into Vectors
 *
 * In more details:
 * It tries to hash entries in the collection using the specified hashing algorithm to build a single vector.
 * If the desired number of features (= hash space size) for all features combined is larger than Integer.Max
 * (the maximal index for a vector), then all the features use the same hash space. There are also options for
 * the user to hash indices with collections where that makes sense (OPLists and OPVectors), and to force a
 * shared hash space, even if the number of feature is not high enough to require it.
 *
 * @param uid uid of the stage
 */
class OPCollectionHashingVectorizer[T <: OPCollection](uid: String = UID[OPCollectionHashingVectorizer[_]])
  (implicit tti: TypeTag[T], val ttvi: TypeTag[T#Value])
  extends SequenceTransformer[T, OPVector](operationName = "vecColHash", uid = uid)
  with VectorizerDefaults with PivotParams with CleanTextFun with HashingFun with HashingVectorizerParams {

  /**
   * Determine if the transformer should use a shared hash space for all features or not
   *
   * @return true if the shared hashing space to be used, false otherwise
   */
  def isSharedHashSpace: Boolean = this.isSharedHashSpace(makeHashingParams())

  /**
   * Get the underlying hashing transformer
   *
   * @return [[HashingTF]]
   */
  def hashingTF(): HashingTF = this.hashingTF(makeHashingParams())

  protected def makeHashingParams() = HashingFunctionParams(
    hashWithIndex = $(hashWithIndex),
    prependFeatureName = $(prependFeatureName),
    numFeatures = $(numFeatures),
    numInputs = inN.length,
    maxNumOfFeatures = TransmogrifierDefaults.MaxNumOfFeatures,
    binaryFreq = $(binaryFreq),
    hashAlgorithm = HashAlgorithm.withNameInsensitive($(hashAlgorithm)),
    hashSpaceStrategy = getHashSpaceStrategy
  )

  override def transformFn: Seq[T] => OPVector = in => hash[T](in, getTransientFeatures(), makeHashingParams())

  /**
   * Function to be called on getMetadata
   */
  override def onGetMetadata(): Unit = {
    val meta = makeVectorMetadata(getTransientFeatures(), makeHashingParams(), getOutputFeatureName)
    setMetadata(meta.toMetadata)
  }
}

private[op] trait HashingVectorizerParams extends Params {
  final val numFeatures = new IntParam(
    parent = this, name = "numFeatures",
    doc = s"number of features (hashes) to generate (default: ${TransmogrifierDefaults.DefaultNumOfFeatures})",
    isValid = ParamValidators.inRange(
      lowerBound = 0, upperBound = TransmogrifierDefaults.MaxNumOfFeatures,
      lowerInclusive = false, upperInclusive = true
    )
  )
  def setNumFeatures(v: Int): this.type = set(numFeatures, v)
  def getNumFeatures(): Int = $(numFeatures)

  final val hashWithIndex = new BooleanParam(
    parent = this, name = "hashWithIndex",
    doc = s"if true, include indices when hashing a feature that has them (OPLists or OPVectors)"
  )
  def setHashWithIndex(v: Boolean): this.type = set(hashWithIndex, v)

  final val hashSpaceStrategy: Param[String] = new Param[String](this, "hashSpaceStrategy",
    "Strategy to determine whether to use shared or separate hash space for input text features",
    (value: String) => HashSpaceStrategy.withNameInsensitiveOption(value).isDefined
  )
  def setHashSpaceStrategy(v: HashSpaceStrategy): this.type = set(hashSpaceStrategy, v.entryName)
  def getHashSpaceStrategy: HashSpaceStrategy = HashSpaceStrategy.withNameInsensitive($(hashSpaceStrategy))

  final val prependFeatureName = new BooleanParam(
    parent = this, name = "prependFeatureName",
    doc = s"if true, prepends a input feature name to each token of that feature"
  )
  def setPrependFeatureName(v: Boolean): this.type = set(prependFeatureName, v)

  final val hashAlgorithm = new Param[String](
    parent = this, name = "hashAlgorithm", doc = s"hash algorithm to use",
    isValid = (s: String) => HashAlgorithm.withNameInsensitiveOption(s).isDefined
  )
  def setHashAlgorithm(h: HashAlgorithm): this.type = set(hashAlgorithm, h.toString.toLowerCase)
  def getHashAlgorithm: HashAlgorithm = HashAlgorithm.withNameInsensitive($(hashAlgorithm))

  final val binaryFreq = new BooleanParam(
    parent = this, name = "binaryFreq",
    doc = "if true, term frequency vector will be binary such that non-zero term counts will be set to 1.0"
  )
  def setBinaryFreq(v: Boolean): this.type = set(binaryFreq, v)

  setDefault(
    numFeatures -> TransmogrifierDefaults.DefaultNumOfFeatures,
    hashWithIndex -> TransmogrifierDefaults.HashWithIndex,
    prependFeatureName -> TransmogrifierDefaults.PrependFeatureName,
    hashAlgorithm -> TransmogrifierDefaults.HashAlgorithm.toString.toLowerCase,
    binaryFreq -> TransmogrifierDefaults.BinaryFreq,
    hashSpaceStrategy -> HashSpaceStrategy.Auto.toString
  )

}

/**
 * Hashing Parameters
 *
 * @param hashWithIndex        if true, include indices when hashing a feature that has them (OPLists or OPVectors)
 * @param prependFeatureName   if true, prepends a input feature name to each token of that feature
 * @param numFeatures          number of features (hashes) to generate
 * @param numInputs            number of inputs
 * @param maxNumOfFeatures     max number of features (hashes)
 * @param binaryFreq           if true, term frequency vector will be binary such that non-zero term counts
 *                             will be set to 1.0
 * @param hashAlgorithm        hash algorithm to use
 * @param hashSpaceStrategy    strategy to determine whether to use shared hash space for all included features
 */
case class HashingFunctionParams
(
  hashWithIndex: Boolean,
  prependFeatureName: Boolean,
  numFeatures: Int,
  numInputs: Int,
  maxNumOfFeatures: Int,
  binaryFreq: Boolean,
  hashAlgorithm: HashAlgorithm,
  hashSpaceStrategy: HashSpaceStrategy
)

/**
 * Hashing functionality
 */
private[op] trait HashingFun {
  self: OpPipelineStageBase =>

  /**
   * Determine if the transformer should use a shared hash space for all features or not
   *
   * @return true if the shared hashing space to be used, false otherwise
   */
  protected def isSharedHashSpace(p: HashingFunctionParams, numFeatures: Option[Int] = None): Boolean = {
    val numHashes = p.numFeatures
    val numOfFeatures = numFeatures.getOrElse(p.numInputs)
    println(">>>>>>>>>>>>>>>>> Total hash space")
    println(numHashes * numOfFeatures)
    println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    p.hashSpaceStrategy match {
      case HashSpaceStrategy.Shared => true
      case HashSpaceStrategy.Separate => false
      case HashSpaceStrategy.Auto => (numHashes * numOfFeatures) > p.maxNumOfFeatures
    }
  }

  /**
   * HashingTF instance
   */
  protected def hashingTF(params: HashingFunctionParams, adaptiveHash: Option[Int] = None): HashingTF = {
    new HashingTF(numFeatures = adaptiveHash.getOrElse(params.numFeatures))
      .setBinary(params.binaryFreq)
      .setHashAlgorithm(params.hashAlgorithm.toString.toLowerCase)
  }

  protected def makeVectorColumnMetadata(
    features: Array[TransientFeature],
    params: HashingFunctionParams,
    hashSizes: Option[Array[Int]]
  ): Array[OpVectorColumnMetadata] = {
    val numFeatures = params.numFeatures
    if (isSharedHashSpace(params)) {
      val allNames = features.map(_.name)
      (0 until numFeatures).map { i =>
        OpVectorColumnMetadata(
          parentFeatureName = features.map(_.name),
          parentFeatureType = features.map(_.typeName),
          grouping = None,
          indicatorValue = None
        )
      }.toArray
    } else {
      hashSizes match {
        case Some(x) =>
          require(x.size == features.size)
          val featureAndSizes = features.zip(x)
          featureAndSizes.flatMap(x => Array.fill(x._2)(x._1.toColumnMetaData()))
        case None =>
          for {
          f <- features
          i <- 0 until numFeatures
          } yield f.toColumnMetaData()
      }
    }
  }

  protected def makeVectorMetadata(
    features: Array[TransientFeature],
    params: HashingFunctionParams,
    outputName: String
  ): OpVectorMetadata = {
    val cols = makeVectorColumnMetadata(features, params, None)
    OpVectorMetadata(outputName, cols, Transmogrifier.inputFeaturesToHistory(features, stageName))
  }

  /**
   * Hashes input sequence of values into OPVector using the supplied hashing params
   */
  protected def hash[T <: OPCollection](
    in: Seq[T],
    features: Array[TransientFeature],
    params: HashingFunctionParams,
    hashSizes: Array[Int]
  ): OPVector = {
    if (in.isEmpty) OPVector.empty
    else {
      val hasher = hashingTF(params)
      val fNameHashesWithInputs = features.map(f => hasher.indexOf(f.name)).zip(in)

      if (isSharedHashSpace(params)) {
        val allElements = ArrayBuffer.empty[Any]
        for {
          (featureNameHash, el) <- fNameHashesWithInputs
          prepared = prepare[T](el, params.hashWithIndex, params.prependFeatureName, featureNameHash)
          p <- prepared
        } allElements.append(p)
        hasher.transform(allElements).asML.toOPVector
      }
      else {
        require(hashSizes.size == features.size)
        val hashers = hashSizes.map(x => hashingTF(params, Some(x)))
        combine(hashers.zip(in).map(
          x => x._1.transform(
            prepare[T](x._2, params.hashWithIndex, params.prependFeatureName, 0)
          ).asML)).toOPVector
      }
    }
  }

  /**
   * Function that prepares the input columns to be hashed
   * Note that MurMur3 hashing algorithm only defined for primitive types so need to convert tuples to strings.
   * MultiPickList sets are hashed as is since there is no meaningful order in the selected choices. Lists
   * and vectors can be hashed with or without their indices, since order may be important. Maps are hashed as
   * (key,value) strings.
   *
   * @param el element we are hashing (eg. an OPList, OPMap, etc.)
   * @return an Iterable object corresponding to the hashed element
   */
  protected def prepare[T <: OPCollection](
    el: T,
    shouldHashWithIndex: Boolean,
    shouldPrependFeatureName: Boolean,
    featureNameHash: Int
  ): Iterable[Any] = {
    val res: Iterable[Any] = el match {
      case el: OPVector =>
        el.value match {
          case d: DenseVector =>
            if (shouldHashWithIndex) d.toArray.zipWithIndex.map(_.toString()) else d.toArray
          case s: SparseVector =>
            val elements = ArrayBuffer.empty[Any]
            s.foreachActive((i, v) => elements.append(if (shouldHashWithIndex) (i, v).toString() else v))
            elements
        }
      case el: OPList[_] => if (shouldHashWithIndex) el.v.zipWithIndex.map(_.toString()) else el.v.map(_.toString())
      case el: OPMap[_] => el.v.map(_.toString())
      case el: OPSet[_] => el.v
    }
    if (shouldPrependFeatureName) res.map(v => s"${featureNameHash}_$v") else res
  }

}

/**
 * Map Hashing functionality
 */
private[op] trait MapHashingFun extends HashingFun {
  self: OpPipelineStageBase =>

  protected def makeVectorColumnMetadata
  (
    hashFeatures: Array[TransientFeature],
    ignoreFeatures: Array[TransientFeature],
    params: HashingFunctionParams,
    hashKeys: Seq[Seq[String]],
    ignoreKeys: Seq[Seq[String]],
    shouldTrackNulls: Boolean,
    shouldTrackLen: Boolean
  ): Array[OpVectorColumnMetadata] = {
    val numHashes = params.numFeatures
    val numFeatures = hashKeys.map(_.length).sum
    val hashColumns =
      if (isSharedHashSpace(params, Some(numFeatures))) {
        (0 until numHashes).map { i =>
          OpVectorColumnMetadata(
            parentFeatureName = hashFeatures.map(_.name),
            parentFeatureType = hashFeatures.map(_.typeName),
            grouping = None,
            indicatorValue = None
          )
        }.toArray
      } else {
        for {
          // Need to filter out empty key sequences since the hashFeatures only contain a map feature if one of their
          // keys is to be hashed, but hashKeys contains a sequence per map (whether it's empty or not)
          (keys, f) <- hashKeys.filter(_.nonEmpty).zip(hashFeatures)
          key <- keys
          i <- 0 until numHashes
        } yield f.toColumnMetaData().copy(grouping = Option(key))
      }.toArray

    // All columns get null tracking or text length tracking, whether their contents are hashed or ignored
    val allTextKeys = hashKeys.zip(ignoreKeys).map{ case(h, i) => h ++ i }
    val allTextFeatures = hashFeatures ++ ignoreFeatures
    val nullColumns = if (shouldTrackNulls) {
      for {
        (keys, f) <- allTextKeys.toArray.zip(allTextFeatures)
        key <- keys
      } yield f.toColumnMetaData(isNull = true).copy(grouping = Option(key))
    } else Array.empty[OpVectorColumnMetadata]

    val lenColumns = if (shouldTrackLen) {
      for {
        (keys, f) <- allTextKeys.toArray.zip(allTextFeatures)
        key <- keys
      } yield f.toColumnMetaData(descriptorValue = OpVectorColumnMetadata.TextLenString).copy(grouping = Option(key))
    } else Array.empty[OpVectorColumnMetadata]

    hashColumns ++ lenColumns ++ nullColumns
  }

  protected def hash
  (
    inputs: Seq[Map[String, TextList]],
    allKeys: Seq[Seq[String]],
    params: HashingFunctionParams
  ): OPVector = {
    if (inputs.isEmpty) OPVector.empty
    else {
      val hasher = hashingTF(params)
      val fNameHashesWithInputsSeq = allKeys.zip(inputs).map{ case (featureKeys, input) =>
        featureKeys.map{ key =>
          val featureHash = hasher.indexOf(key)
          featureHash -> input.getOrElse(key, TextList.empty)
        }
      }

      val numFeatures = allKeys.map(_.length).sum
      if (isSharedHashSpace(params, Some(numFeatures))) {
        val allElements = ArrayBuffer.empty[Any]
        for{
          fNameHashesWithInputs <- fNameHashesWithInputsSeq
          (featureNameHash, values) <- fNameHashesWithInputs
          prepared = prepare[TextList](values, params.hashWithIndex, params.prependFeatureName, featureNameHash)
          p <- prepared
        } allElements.append(p)

        hasher.transform(allElements).asML.toOPVector
      } else {
        val hashedVecs =
          fNameHashesWithInputsSeq.map(_.map { case (featureNameHash, el) =>
            hasher.transform(
              prepare[TextList](el, params.hashWithIndex, params.prependFeatureName, featureNameHash)
            ).asML
          })
        combine(hashedVecs.flatten).toOPVector
      }
    }
  }
}
