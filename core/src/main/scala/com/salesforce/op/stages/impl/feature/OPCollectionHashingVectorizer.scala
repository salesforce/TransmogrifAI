/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamValidators}
import org.apache.spark.mllib.feature.HashingTF

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe.TypeTag

/**
 * Generic hashing vectorizer to convert features of type OPCollection into Vectors
 *
 * In more details:
 * It tries to hash entries in the collection using the specified hashing algorithm to build a single vector.
 * If the desired number of features (= hash space size) for all features combines is larger than Integer.Max
 * (the maximal index for a vector), then all the features use the same hash space. There are also options for
 * the user to hash indices with collections where that makes sense (OPLists and OPVectors), and to force a
 * shared hash space, even if the number of feature is not high enough to require it.
 *
 * @param uid uid of the stage
 */
class OPCollectionHashingVectorizer[T <: OPCollection](uid: String = UID[OPCollectionHashingVectorizer[_]])
  (implicit tti: TypeTag[T], val ttvi: TypeTag[T#Value])
  extends SequenceTransformer[T, OPVector](operationName = "vecColHash", uid = uid)
  with VectorizerDefaults with PivotParams with CleanTextFun {

  final val numFeatures = new IntParam(
    parent = this, name = "numFeatures",
    doc = s"number of features (hashes) to generate (default: ${Transmogrifier.DefaultNumOfFeatures})",
    isValid = ParamValidators.inRange(
      lowerBound = 0, upperBound = Transmogrifier.MaxNumOfFeatures, lowerInclusive = false, upperInclusive = true
    )
  )
  def setNumFeatures(v: Int): this.type = set(numFeatures, v)
  def getNumFeatures(): Int = $(numFeatures)

  final val hashWithIndex = new BooleanParam(
    parent = this, name = "hashWithIndex",
    doc = s"if true, include indices when hashing a feature that has them (OPLists or OPVectors)"
  )
  def setHashWithIndex(v: Boolean): this.type = set(hashWithIndex, v)

  final val forceSharedHashSpace = new BooleanParam(
    parent = this, name = "forceSharedHashSpace",
    doc = s"if true, then force the hash space to be shared among all included features"
  )
  def setForceSharedHashSpace(v: Boolean): this.type = set(forceSharedHashSpace, v)

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

  final val binaryFreq = new BooleanParam(
    parent = this, name = "binaryFreq",
    doc = "if true, term frequency vector will be binary such that non-zero term counts will be set to 1.0"
  )
  def setBinaryFreq(v: Boolean): this.type = set(binaryFreq, v)

  setDefault(
    numFeatures -> Transmogrifier.DefaultNumOfFeatures,
    hashWithIndex -> Transmogrifier.HashWithIndex,
    forceSharedHashSpace -> false,
    prependFeatureName -> Transmogrifier.PrependFeatureName,
    hashAlgorithm -> Transmogrifier.HashAlgorithm.toString.toLowerCase,
    binaryFreq -> Transmogrifier.BinaryFreq
  )

  /**
   * Determine if the transformer should use a shared hash space for all features or not
   *
   * @return true if the shared hashing space to be used, false otherwise
   */
  def isSharedHashSpace: Boolean =
    (getNumFeatures() * inN.length) > Transmogrifier.MaxNumOfFeatures || $(forceSharedHashSpace)

  /**
   * Get the underlying hashing transformer
   *
   * @return
   */
  def hashingTF(): HashingTF = {
    new HashingTF(numFeatures = $(numFeatures))
      .setBinary($(binaryFreq))
      .setHashAlgorithm($(hashAlgorithm))
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
  private def prepare(
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

  /**
   * Function used to convert input to output
   */
  override def transformFn: Seq[T] => OPVector = in => {
    if (in.isEmpty) OPVector.empty
    else {
      val hasher = hashingTF()
      val shouldHashWithIndex = $(hashWithIndex)
      val shouldPrependFeatureName = $(prependFeatureName)
      val fNameHashesWithInputs = getTransientFeatures().map(f => hasher.indexOf(f.name)).zip(in)

      if (isSharedHashSpace) {
        val allElements = ArrayBuffer.empty[Any]
        for {
          (featureNameHash, el) <- fNameHashesWithInputs
          prepared = prepare(el, shouldHashWithIndex, shouldPrependFeatureName, featureNameHash)
          p <- prepared
        } allElements.append(p)

        hasher.transform(allElements).asML.toOPVector
      }
      else {
        val hashedVecs =
          fNameHashesWithInputs.map { case (featureNameHash, el) =>
            hasher.transform(prepare(el, shouldHashWithIndex, shouldPrependFeatureName, featureNameHash)).asML
          }
        VectorsCombiner.combine(hashedVecs).toOPVector
      }
    }
  }

  /**
   * Function to be called on getMetadata
   */
  override protected def onGetMetadata(): Unit = {
    val numFeatures = getNumFeatures()
    val cols =
      if (isSharedHashSpace) {
        val allNames = inN.map(_.name)
        (0 until numFeatures).map { i =>
          OpVectorColumnMetadata(
            parentFeatureName = inN.map(_.name),
            parentFeatureType = inN.map(_.typeName),
            indicatorGroup = None,
            indicatorValue = None
          )
        }.toArray
      } else {
        for {
          f <- inN
          i <- 0 until numFeatures
        } yield OpVectorColumnMetadata(
          parentFeatureName = Seq(f.name),
          parentFeatureType = Seq(f.typeName),
          indicatorGroup = None,
          indicatorValue = None
        )
      }
    setMetadata(OpVectorMetadata(outputName, cols, Transmogrifier.inputFeaturesToHistory(inN)).toMetadata)
  }
}
