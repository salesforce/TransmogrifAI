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

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.{BinaryMap, _}
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature._
import com.salesforce.op.utils.text.Language
import org.apache.spark.ml.linalg.Vectors

import scala.reflect.runtime.universe._

trait RichMapFeature {

  /**
   * Enrichment functions for OPMap Features
   *
   * @param f FeatureLike
   */
  implicit class RichMapFeature[T <: OPMap[_] : TypeTag](val f: FeatureLike[T]) {

    /**
     * Filters map by whitelisted and blacklisted keys
     *
     * @param whiteList whitelisted keys
     * @param blackList blacklisted keys
     * @return filtered OPMap feature
     */
    def filter(whiteList: Seq[String], blackList: Seq[String]): FeatureLike[T] = {
      f.transformWith(
        new FilterMap[T]()
          .setWhiteListKeys(whiteList.toSet.toArray)
          .setBlackListKeys(blackList.toSet.toArray)
      )
    }
  }

  /**
   * Enrichment functions for OPMap Features with String values. All are pivoted by default except TextMap and
   * TextAreaMap which are defined specially below.
   *
   * @param f FeatureLike
   */
  implicit class RichStringMapFeature[T <: OPMap[String] : TypeTag](val f: FeatureLike[T])
    (implicit val ttiv: TypeTag[T#Value]) {

    /**
     * Apply TextMapPivotVectorizer on any OPMap that has string values
     *
     * @param others            other features of the same type
     * @param topK              number of values to keep for each key
     * @param minSupport        min times a value must occur to be retained in pivot
     * @param cleanText         clean text before pivoting
     * @param cleanKeys         clean map keys before pivoting
     * @param whiteListKeys     keys to whitelist
     * @param blackListKeys     keys to blacklist
     * @param trackNulls        option to keep track of values that were missing
     * @param maxPctCardinality max percentage of distinct values a categorical feature can have (between 0.0 and 1.00)
     *
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      others: Array[FeatureLike[T]] = Array.empty,
      maxPctCardinality: Double = OpOneHotVectorizer.MaxPctCardinality
    ): FeatureLike[OPVector] = {
      new TextMapPivotVectorizer[T]()
        .setInput(f +: others)
        .setTopK(topK)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setMinSupport(minSupport)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .setMaxPercentageCardinality(maxPctCardinality)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for Base64Map features.
   *
   * @param f FeatureLike
   */
  implicit class RichBase64MapFeature(val f: FeatureLike[Base64Map]) {

    /**
     * Detect MIME type for Base64Map encoded binary data
     *
     * @param typeHint MIME type hint, i.e. 'application/json', 'text/plain' etc.
     * @return mime type as text
     */
    def detectMimeTypes(typeHint: Option[String] = None): FeatureLike[PickListMap] = {
      val detector = new MimeTypeMapDetector()
      typeHint.foreach(detector.setTypeHint)
      f.transformWith(detector)
    }

    /**
     * Base64Map vecrtorization:
     * MIME types are extracted, and the maps are converted into PickListMaps
     * and then vectorized using the TextMapPivotVectorizer.
     *
     *
     * @param others            other features of the same type
     * @param topK              number of values to keep for each key
     * @param minSupport        min times a value must occur to be retained in pivot
     * @param cleanText         clean text before pivoting
     * @param cleanKeys         clean map keys before pivoting
     * @param whiteListKeys     keys to whitelist
     * @param blackListKeys     keys to blacklist
     * @param typeHint          optional hint for MIME type detector
     * @param trackNulls        option to keep track of values that were missing
     * @param maxPctCardinality max percentage of distinct values a categorical feature can have (between 0.0 and 1.00)
     *
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      typeHint: Option[String] = None,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      others: Array[FeatureLike[Base64Map]] = Array.empty,
      maxPctCardinality: Double = OpOneHotVectorizer.MaxPctCardinality
    ): FeatureLike[OPVector] = {

      val feats: Array[FeatureLike[PickListMap]] = (f +: others).map(_.detectMimeTypes(typeHint))

      new TextMapPivotVectorizer[PickListMap]()
        .setInput(feats)
        .setTopK(topK)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setMinSupport(minSupport)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .setMaxPercentageCardinality(maxPctCardinality)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for TextMap Features (they are hashed by default instead of being pivoted)
   *
   * @param f FeatureLike
   */
  implicit class RichTextMapFeature(val f: FeatureLike[TextMap]) {

    /**
     * Apply TextMapVectorizer on any OPMap that has string values
     *
     * @param others                   other features of the same type
     * @param cleanText                clean text before pivoting
     * @param cleanKeys                clean map keys before pivoting
     * @param shouldPrependFeatureName whether or not to prepend feature name hash to the tokens before hashing
     * @param whiteListKeys            keys to whitelist
     * @param blackListKeys            keys to blacklist
     * @param trackNulls               option to keep track of values that were missing
     * @param trackTextLen             option to add a column containing the text length to the feature vector
     * @param numHashes                size of hash space
     * @param hashSpaceStrategy        strategy to determine whether to use shared hash space for all included features
     *
     * @return an OPVector feature
     */
    def vectorize(
      cleanText: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      shouldPrependFeatureName: Boolean = TransmogrifierDefaults.PrependFeatureName,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[TextMap]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      trackTextLen: Boolean = TransmogrifierDefaults.TrackTextLen,
      numHashes: Int = TransmogrifierDefaults.DefaultNumOfFeatures,
      hashSpaceStrategy: HashSpaceStrategy = TransmogrifierDefaults.HashSpaceStrategy
    ): FeatureLike[OPVector] = {
      val hashedFeatures = new TextMapHashingVectorizer[TextMap]()
        .setInput(f +: others)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setPrependFeatureName(shouldPrependFeatureName)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(false) // Null tracking does nothing here and is done from outside the vectorizer, below
        .setNumFeatures(numHashes)
        .setHashSpaceStrategy(hashSpaceStrategy)
        .getOutput()

      /**
       * Note: Text is tokenized into a TextList, and then null tracking is applied. For maps, we do null
       * tracking on the original features so it's slightly different. Fortunately, tokenization for TextMaps is done
       * via the tokenize function directly, rather than with an entire stage, so things should still work here.
       */
      (trackTextLen, trackNulls) match {
        case (true, true) =>
          val textLengths = new TextMapLenEstimator[TextMap]().setInput(f +: others).getOutput()
          val nullIndicators = new TextMapNullEstimator[TextMap]().setInput(f +: others).getOutput()
          new VectorsCombiner().setInput(hashedFeatures, textLengths, nullIndicators).getOutput()
        case (true, false) =>
          val textLengths = new TextMapLenEstimator[TextMap]().setInput(f +: others).getOutput()
          new VectorsCombiner().setInput(hashedFeatures, textLengths).getOutput()
        case (false, true) =>
          val nullIndicators = new TextMapNullEstimator[TextMap]().setInput(f +: others).getOutput()
          new VectorsCombiner().setInput(hashedFeatures, nullIndicators).getOutput()
        case(false, false) => hashedFeatures
      }
    }

    /**
     * Vectorize text map features by treating low cardinality text features as categoricals and
     * applying hashing trick to high caridinality ones.
     *
     * @param maxCategoricalCardinality max cardinality for a text feature to be treated as categorical
     * @param numHashes                 number of features (hashes) to generate
     * @param autoDetectLanguage        indicates whether to attempt language detection
     * @param minTokenLength            minimum token length, >= 1.
     * @param toLowercase               indicates whether to convert all characters to lowercase before analyzing
     * @param cleanText                 indicates whether to ignore capitalization and punctuation
     * @param cleanKeys                 clean map keys before pivoting
     * @param trackNulls                indicates whether or not to track null values in a separate column.
     * @param trackTextLen              indicates whether or not to track the length of the text in a separate column
     * @param topK                      number of most common elements to be used as categorical pivots
     * @param minSupport                minimum number of occurrences an element must have to appear in pivot
     * @param unseenName                name to give indexes which do not have a label name associated with them
     * @param hashWithIndex             include indices when hashing a feature that has them (OPLists or OPVectors)
     * @param binaryFreq                if true, term frequency vector will be binary such that non-zero term
     *                                  counts will be set to 1.0
     * @param prependFeatureName        if true, prepends a input feature name to each token of that feature
     * @param autoDetectThreshold       Language detection threshold. If none of the detected languages have
     *                                  confidence greater than the threshold then defaultLanguage is used.
     * @param hashSpaceStrategy         strategy to determine whether to use shared hash space for all included features
     * @param defaultLanguage           default language to assume in case autoDetectLanguage is disabled or
     *                                  failed to make a good enough prediction.
     * @param hashAlgorithm             hash algorithm to use
     * @param others                    additional text features
     * @return result feature of type Vector
     */
    // scalastyle:off parameter.number
    def smartVectorize
    (
      maxCategoricalCardinality: Int,
      numHashes: Int,
      autoDetectLanguage: Boolean,
      minTokenLength: Int,
      toLowercase: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      cleanText: Boolean = TransmogrifierDefaults.CleanText,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      trackTextLen: Boolean = TransmogrifierDefaults.TrackTextLen,
      topK: Int = TransmogrifierDefaults.TopK,
      minSupport: Int = TransmogrifierDefaults.MinSupport,
      unseenName: String = TransmogrifierDefaults.OtherString,
      hashWithIndex: Boolean = TransmogrifierDefaults.HashWithIndex,
      binaryFreq: Boolean = TransmogrifierDefaults.BinaryFreq,
      prependFeatureName: Boolean = TransmogrifierDefaults.PrependFeatureName,
      autoDetectThreshold: Double = TextTokenizer.AutoDetectThreshold,
      hashSpaceStrategy: HashSpaceStrategy = TransmogrifierDefaults.HashSpaceStrategy,
      defaultLanguage: Language = TextTokenizer.DefaultLanguage,
      hashAlgorithm: HashAlgorithm = TransmogrifierDefaults.HashAlgorithm,
      others: Array[FeatureLike[TextMap]] = Array.empty
    ): FeatureLike[OPVector] = {
      // scalastyle:on parameter.number
      new SmartTextMapVectorizer[TextMap]()
        .setInput(f +: others)
        .setMaxCardinality(maxCategoricalCardinality)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setTrackNulls(trackNulls)
        .setTrackTextLen(trackTextLen)
        .setAutoDetectLanguage(autoDetectLanguage)
        .setAutoDetectThreshold(autoDetectThreshold)
        .setDefaultLanguage(defaultLanguage)
        .setMinTokenLength(minTokenLength)
        .setToLowercase(toLowercase)
        .setTopK(topK)
        .setMinSupport(minSupport)
        .setUnseenName(unseenName)
        .setNumFeatures(numHashes)
        .setHashWithIndex(hashWithIndex)
        .setPrependFeatureName(prependFeatureName)
        .setHashSpaceStrategy(hashSpaceStrategy)
        .setHashAlgorithm(hashAlgorithm)
        .setBinaryFreq(binaryFreq)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for TextAreaMap Features (they are hashed by default instead of being pivoted)
   *
   * @param f FeatureLike
   */
  implicit class RichTextAreaMapFeature(val f: FeatureLike[TextAreaMap]) {

    /**
     * Apply TextMapVectorizer on any OPMap that has string values
     *
     * @param others                   other features of the same type
     * @param cleanText                clean text before pivoting
     * @param cleanKeys                clean map keys before pivoting
     * @param shouldPrependFeatureName whether or not to prepend feature name hash to the tokens before hashing
     * @param whiteListKeys            keys to whitelist
     * @param blackListKeys            keys to blacklist
     * @param trackNulls               option to keep track of values that were missing
     * @param trackTextLen             option to keep track of text lengths
     * @param numHashes                size of hash space
     * @param hashSpaceStrategy        strategy to determine whether to use shared hash space for all included features
     *
     * @return an OPVector feature
     */
    def vectorize(
      cleanText: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      shouldPrependFeatureName: Boolean = TransmogrifierDefaults.PrependFeatureName,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[TextAreaMap]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      trackTextLen: Boolean = TransmogrifierDefaults.TrackTextLen,
      numHashes: Int = TransmogrifierDefaults.DefaultNumOfFeatures,
      hashSpaceStrategy: HashSpaceStrategy = TransmogrifierDefaults.HashSpaceStrategy
    ): FeatureLike[OPVector] = {
      val hashedFeatures = new TextMapHashingVectorizer[TextAreaMap]()
        .setInput(f +: others)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setPrependFeatureName(shouldPrependFeatureName)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(false) // Null tracking does nothing here and is done from outside the vectorizer, below
        .setNumFeatures(numHashes)
        .setHashSpaceStrategy(hashSpaceStrategy)
        .getOutput()

      /* Note: Text is tokenized into a TextList, and then null tracking is applied. For maps, we do null
        tracking on the original features so it's slightly different. Fortunately, tokenization for TextMaps is done
        via the tokenize function directly, rather than with an entire stage, so things should still work here.
       */
      (trackTextLen, trackNulls) match {
        case (true, true) =>
          val textLengths = new TextMapLenEstimator[TextAreaMap]().setInput(f +: others).getOutput()
          val nullIndicators = new TextMapNullEstimator[TextAreaMap]().setInput(f +: others).getOutput()
          new VectorsCombiner().setInput(hashedFeatures, textLengths, nullIndicators).getOutput()
        case (true, false) =>
          val textLengths = new TextMapLenEstimator[TextAreaMap]().setInput(f +: others).getOutput()
          new VectorsCombiner().setInput(hashedFeatures, textLengths).getOutput()
        case (false, true) =>
          val nullIndicators = new TextMapNullEstimator[TextAreaMap]().setInput(f +: others).getOutput()
          new VectorsCombiner().setInput(hashedFeatures, nullIndicators).getOutput()
        case(false, false) => hashedFeatures
      }
    }

    /**
     * Vectorize textarea map features by treating low cardinality text features as categoricals and
     * applying hashing trick to high caridinality ones.
     *
     * @param maxCategoricalCardinality max cardinality for a text feature to be treated as categorical
     * @param numHashes                 number of features (hashes) to generate
     * @param autoDetectLanguage        indicates whether to attempt language detection
     * @param minTokenLength            minimum token length, >= 1.
     * @param toLowercase               indicates whether to convert all characters to lowercase before analyzing
     * @param cleanKeys                 clean map keys before pivoting
     * @param cleanText                 indicates whether to ignore capitalization and punctuation
     * @param trackNulls                indicates whether or not to track null values in a separate column.
     * @param trackTextLen              indicates whether or not to track the length of the text in a separate column
     * @param topK                      number of most common elements to be used as categorical pivots
     * @param minSupport                minimum number of occurrences an element must have to appear in pivot
     * @param unseenName                name to give indexes which do not have a label name associated with them
     * @param hashWithIndex             include indices when hashing a feature that has them (OPLists or OPVectors)
     * @param binaryFreq                if true, term frequency vector will be binary such that non-zero term
     *                                  counts will be set to 1.0
     * @param prependFeatureName        if true, prepends a input feature name to each token of that feature
     * @param autoDetectThreshold       Language detection threshold. If none of the detected languages have
     *                                  confidence greater than the threshold then defaultLanguage is used.
     * @param hashSpaceStrategy         strategy to determine whether to use shared hash space for all included features
     * @param defaultLanguage           default language to assume in case autoDetectLanguage is disabled or
     *                                  failed to make a good enough prediction.
     * @param hashAlgorithm             hash algorithm to use
     * @param others                    additional text features
     * @return result feature of type Vector
     */
    // scalastyle:off parameter.number
    def smartVectorize
    (
      maxCategoricalCardinality: Int,
      numHashes: Int,
      autoDetectLanguage: Boolean,
      minTokenLength: Int,
      toLowercase: Boolean,
      cleanText: Boolean = TransmogrifierDefaults.CleanText,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      trackTextLen: Boolean = TransmogrifierDefaults.TrackTextLen,
      topK: Int = TransmogrifierDefaults.TopK,
      minSupport: Int = TransmogrifierDefaults.MinSupport,
      unseenName: String = TransmogrifierDefaults.OtherString,
      hashWithIndex: Boolean = TransmogrifierDefaults.HashWithIndex,
      binaryFreq: Boolean = TransmogrifierDefaults.BinaryFreq,
      prependFeatureName: Boolean = TransmogrifierDefaults.PrependFeatureName,
      autoDetectThreshold: Double = TextTokenizer.AutoDetectThreshold,
      hashSpaceStrategy: HashSpaceStrategy = TransmogrifierDefaults.HashSpaceStrategy,
      defaultLanguage: Language = TextTokenizer.DefaultLanguage,
      hashAlgorithm: HashAlgorithm = TransmogrifierDefaults.HashAlgorithm,
      others: Array[FeatureLike[TextAreaMap]] = Array.empty
    ): FeatureLike[OPVector] = {
      // scalastyle:on parameter.number
      new SmartTextMapVectorizer[TextAreaMap]()
        .setInput(f +: others)
        .setMaxCardinality(maxCategoricalCardinality)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setTrackNulls(trackNulls)
        .setTrackTextLen(trackTextLen)
        .setAutoDetectLanguage(autoDetectLanguage)
        .setAutoDetectThreshold(autoDetectThreshold)
        .setDefaultLanguage(defaultLanguage)
        .setMinTokenLength(minTokenLength)
        .setToLowercase(toLowercase)
        .setTopK(topK)
        .setMinSupport(minSupport)
        .setUnseenName(unseenName)
        .setNumFeatures(numHashes)
        .setHashWithIndex(hashWithIndex)
        .setPrependFeatureName(prependFeatureName)
        .setHashSpaceStrategy(hashSpaceStrategy)
        .setHashAlgorithm(hashAlgorithm)
        .setBinaryFreq(binaryFreq)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for OPMap Features with String values
   *
   * @param f FeatureLike
   */
  implicit class RichMultiPickListMapFeature[T <: OPMap[Set[String]] : TypeTag](val f: FeatureLike[T])
    (implicit val ttiv: TypeTag[T#Value]) {

    /**
     * Apply MultiPickListMapVectorizer on any OPMap that has set values
     *
     * @param others            other features of the same type
     * @param topK              number of values to keep for each key
     * @param minSupport        min times a value must occur to be retained in pivot
     * @param cleanText         clean text before pivoting
     * @param cleanKeys         clean map keys before pivoting
     * @param whiteListKeys     keys to whitelist
     * @param blackListKeys     keys to blacklist
     * @param trackNulls        option to keep track of values that were missing
     * @param maxPctCardinality max percentage of distinct values a categorical feature can have (between 0.0 and 1.00)
     *
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      maxPctCardinality: Double = OpOneHotVectorizer.MaxPctCardinality
    ): FeatureLike[OPVector] = {
      new MultiPickListMapVectorizer[T]()
        .setInput(f +: others)
        .setTopK(topK)
        .setMinSupport(minSupport)
        .setCleanText(cleanText)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .setMaxPercentageCardinality(maxPctCardinality)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for OPMap Features with Double values
   *
   * @param f FeatureLike
   */
  implicit class RichRealMapFeature[T <: OPMap[Double] : TypeTag](val f: FeatureLike[T])
    (implicit val ttiv: TypeTag[T#Value]) {

    /**
     * Apply a smart bucketizer transformer
     *
     * @param label         label feature
     * @param trackNulls    option to keep track of values that were missing
     * @param trackInvalid  option to keep track of invalid values,
     *                      eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain   minimum info gain, one of the stopping criteria of the Decision Tree
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     */
    def autoBucketize(
      label: FeatureLike[RealNN],
      trackNulls: Boolean,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = DecisionTreeNumericBucketizer.MinInfoGain,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty
    ): FeatureLike[OPVector] = {
      new DecisionTreeNumericMapBucketizer[Double, T]()
        .setInput(label, f)
        .setTrackInvalid(trackInvalid)
        .setTrackNulls(trackNulls)
        .setMinInfoGain(minInfoGain)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys).getOutput()
    }

    /**
     * Apply RealMapVectorizer or auto bucketizer (when label is present) on any OPMap that has double values
     *
     * @param others        other features of the same type
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     * @param label         optional label column to be passed into autoBucketizer if present
     * @param trackInvalid  option to keep track of invalid values,
     *                      eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain   minimum info gain, one of the stopping criteria of the Decision Tree
     *
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      fillWithMean: Boolean = TransmogrifierDefaults.FillWithMean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = TransmogrifierDefaults.MinInfoGain,
      label: Option[FeatureLike[RealNN]] = None
    ): FeatureLike[OPVector] = {
      label match {
        case None =>
          new RealMapVectorizer[T]()
            .setInput(f +: others)
            .setFillWithMean(fillWithMean)
            .setDefaultValue(defaultValue)
            .setCleanKeys(cleanKeys)
            .setWhiteListKeys(whiteListKeys)
            .setBlackListKeys(blackListKeys)
            .setTrackNulls(trackNulls)
            .getOutput()
        case Some(lbl) =>
          autoBucketize(
            label = lbl, trackNulls = trackNulls, trackInvalid = trackInvalid,
            minInfoGain = minInfoGain, cleanKeys = cleanKeys,
            whiteListKeys = whiteListKeys, blackListKeys = blackListKeys
          )
      }
    }
  }

  /**
   * Enrichment functions for OPMap Features with Long values
   *
   * @param f FeatureLike
   */
  implicit class RichIntegralMapFeature[T <: OPMap[Long] : TypeTag](val f: FeatureLike[T])
    (implicit val ttiv: TypeTag[T#Value]) {

    /**
     * Apply a smart bucketizer transformer
     *
     * @param label         label feature
     * @param trackNulls    option to keep track of values that were missing
     * @param trackInvalid  option to keep track of invalid values,
     *                      eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain   minimum info gain, one of the stopping criteria of the Decision Tree
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     */
    def autoBucketize(
      label: FeatureLike[RealNN],
      trackNulls: Boolean,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = DecisionTreeNumericBucketizer.MinInfoGain,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty
    ): FeatureLike[OPVector] = {
      new DecisionTreeNumericMapBucketizer[Long, T]()
        .setInput(label, f)
        .setTrackInvalid(trackInvalid)
        .setTrackNulls(trackNulls)
        .setMinInfoGain(minInfoGain)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys).getOutput()
    }

    /**
     * Apply IntegralMapVectorizer or auto bucketizer (when label is present) on any OPMap that has long values
     *
     * @param others        other features of the same type
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     * @param label         optional label column to be passed into autoBucketizer if present
     * @param trackInvalid  option to keep track of invalid values,
     *                      eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain   minimum info gain, one of the stopping criteria of the Decision Tree
     *
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      fillWithMode: Boolean = TransmogrifierDefaults.FillWithMode,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = TransmogrifierDefaults.MinInfoGain,
      label: Option[FeatureLike[RealNN]] = None
    ): FeatureLike[OPVector] = {
      label match {
        case None =>
          new IntegralMapVectorizer[T]()
            .setInput(f +: others)
            .setFillWithMode(fillWithMode)
            .setDefaultValue(defaultValue)
            .setCleanKeys(cleanKeys)
            .setWhiteListKeys(whiteListKeys)
            .setBlackListKeys(blackListKeys)
            .setTrackNulls(trackNulls)
            .getOutput()
        case Some(lbl) =>
          autoBucketize(
            label = lbl, trackNulls = trackNulls, trackInvalid = trackInvalid,
            minInfoGain = minInfoGain, cleanKeys = cleanKeys,
            whiteListKeys = whiteListKeys, blackListKeys = blackListKeys
          )
      }
    }
  }

  /**
   * Enrichment functions for OPMap Features with Date values
   *
   * @param f FeatureLike
   */
  implicit class RichDateMapFeature(val f: FeatureLike[DateMap]) {

    /**
     * transforms a DateMap field into a series of cartesian coordinate representation
     * of an extracted time period on the unit circle
     *
     * @param timePeriod The time period to extract from the timestamp
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param others     Other features of same type
     * enum from: DayOfMonth, DayOfWeek, DayOfYear, HourOfDay, WeekOfMonth, WeekOfYear
     */
    def toUnitCircle
    (
      timePeriod: TimePeriod = TimePeriod.HourOfDay,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[DateMap]] = Array.empty
    ): FeatureLike[OPVector] = {
      new DateMapToUnitCircleVectorizer[DateMap]()
        .setInput(f +: others)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTimePeriod(timePeriod)
        .getOutput()
    }


    /**
     * Apply DateMapVectorizer on any OPMap that has long values
     *
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     * @param referenceDate reference date to subtract off before converting to vector
     * @param circularDateReps list of all the circular date representations that should be included in feature vector
     * @return result feature of type Vector
     * @param others        other features of the same type
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      referenceDate: org.joda.time.DateTime = TransmogrifierDefaults.ReferenceDate,
      circularDateReps: Seq[TimePeriod] = TransmogrifierDefaults.CircularDateRepresentations,
      others: Array[FeatureLike[DateMap]] = Array.empty
    ): FeatureLike[OPVector] = {

      val timePeriods = circularDateReps.map {
        tp => f.toUnitCircle(tp, cleanKeys, whiteListKeys, blackListKeys, others)
      }

      val time = new DateMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .setReferenceDate(referenceDate)
        .getOutput()

      if (timePeriods.isEmpty) time else (timePeriods :+ time).combine()
    }
  }

  /**
   * Enrichment functions for OPMap Features with DateTime values
   *
   * @param f FeatureLike
   */
  implicit class RichDateTimeMapFeature(val f: FeatureLike[DateTimeMap]) {


    /**
     * transforms a DateTimeMap field into a series of cartesian coordinate representation
     * of an extracted time period on the unit circle
     *
     * @param timePeriod The time period to extract from the timestamp
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param others     Other features of same type
     * enum from: DayOfMonth, DayOfWeek, DayOfYear, HourOfDay, WeekOfMonth, WeekOfYear
     */
    def toUnitCircle
    (
      timePeriod: TimePeriod = TimePeriod.HourOfDay,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[DateTimeMap]] = Array.empty
    ): FeatureLike[OPVector] = {
      new DateMapToUnitCircleVectorizer[DateTimeMap]()
        .setInput(f +: others)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTimePeriod(timePeriod)
        .getOutput()
    }

    /**
     * Apply DateMapVectorizer on any OPMap that has long values
     *
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     * @param referenceDate reference date to subtract off before converting to vector
     * @param circularDateReps list of all the circular date representations that should be included in feature vector
     * @param others        other features of the same type
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      referenceDate: org.joda.time.DateTime = TransmogrifierDefaults.ReferenceDate,
      circularDateReps: Seq[TimePeriod] = TransmogrifierDefaults.CircularDateRepresentations,
      others: Array[FeatureLike[DateTimeMap]] = Array.empty
    ): FeatureLike[OPVector] = {

      val timePeriods = circularDateReps.map {
        tp => f.toUnitCircle(tp, cleanKeys, whiteListKeys, blackListKeys, others)
      }

      val time = new DateMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .setReferenceDate(referenceDate)
        .getOutput()

      if (timePeriods.isEmpty) time else (timePeriods :+ time).combine()
    }
  }

  /**
   * Enrichment functions for OPMap Features with Boolean values
   *
   * @param f FeatureLike
   */
  implicit class RichBooleanMapFeature(val f: FeatureLike[BinaryMap]) {

    /**
     * Apply IntegralMapVectorizer on any OPMap that has boolean values
     *
     * @param others        other features of the same type
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     *
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[BinaryMap]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
    ): FeatureLike[OPVector] = {
      new BinaryMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for OPMap Features with Geolocation values
   *
   * @param f FeatureLike
   */
  implicit class RichGeolocationMapFeature(val f: FeatureLike[GeolocationMap]) {

    /**
     * Apply GeolocationMapVectorizer on OPMap that has Geolocation values
     *
     * @param others        other features of the same type
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     *
     * @return an OPVector feature
     */
    def vectorize(
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      defaultValue: Geolocation = TransmogrifierDefaults.DefaultGeolocation,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[GeolocationMap]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
    ): FeatureLike[OPVector] = {
      new GeolocationMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for PhoneMap features
   *
   * @param f FeatureLike
   */
  implicit class RichPhoneMapFeature(val f: FeatureLike[PhoneMap]) {

    /**
     * Returns new feature where true represents valid numbers and false represents invalid numbers
     *
     * @param isStrict      strict comparison if true.
     * @param defaultRegion default locale if region code is not valid
     * @return result feature of type Binary
     */
    def isValidPhoneDefaultCountryMap
    (
      isStrict: Boolean = PhoneNumberParser.StrictValidation,
      defaultRegion: String = PhoneNumberParser.DefaultRegion
    ): FeatureLike[BinaryMap] = {
      f.transformWith(
        new IsValidPhoneMapDefaultCountry()
          .setStrictness(isStrict)
          .setDefaultRegion(defaultRegion)
      )
    }

    /**
     * Returns a vector for phone numbers where the first element is 1 if the number is valid for the given region
     * 0 if invalid and with an optional second element idicating if the phone number was null
     *
     * @param defaultRegion region against which to check phone validity
     * @param isStrict      strict validation means cannot have extra digits
     * @param fillValue     value to fill in for nulls in vactor creation
     * @param others        other phone numbers to vectorize
     * @param trackNulls    option to keep track of values that were missing
     * @return vector feature containing information about phone number
     */
    def vectorize(
      defaultRegion: String,
      isStrict: Boolean = PhoneNumberParser.StrictValidation,
      fillValue: Double = TransmogrifierDefaults.FillValue,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[PhoneMap]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
    ): FeatureLike[OPVector] = {
      val valid = (f +: others).map(_.isValidPhoneDefaultCountryMap(defaultRegion = defaultRegion, isStrict = isStrict))
      valid.head.vectorize(others = valid.tail, defaultValue = fillValue, trackNulls = trackNulls)
    }
  }

  /**
   * Enrichment functions for EmailMap Features
   *
   * @param f FeatureLike of EmailMap
   */
  implicit class RichEmailMapFeature(val f: FeatureLike[EmailMap]) {

    /**
     * Transform EmailMap feature to PickListMap by extracting email domains, converting them
     * to PickList and then vectorize the PickListMap
     *
     * @param topK              number of values to keep for each key
     * @param minSupport        min times a value must occur to be retained in pivot
     * @param cleanText         clean text after email split but before pivoting
     * @param cleanKeys         clean map keys before pivoting
     * @param whiteListKeys     keys to whitelist
     * @param blackListKeys     keys to blacklist
     * @param trackNulls        option to keep track of values that were missing
     * @param maxPctCardinality max percentage of distinct values a categorical feature can have (between 0.0 and 1.00)
     *
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      others: Array[FeatureLike[EmailMap]] = Array.empty,
      maxPctCardinality: Double = OpOneHotVectorizer.MaxPctCardinality
    ): FeatureLike[OPVector] = {
      val domains: Array[FeatureLike[PickListMap]] = (f +: others).map { e =>
        val transformer = new OPMapTransformer[Email, PickList, EmailMap, PickListMap](
          operationName = "emailToPickListMap",
          transformer = new UnaryLambdaTransformer[Email, PickList](
            operationName = "emailToPickList",
            transformFn = _.domain.toPickList
          )
        )
        transformer.setInput(e).getOutput()
      }

      domains.head.vectorize(
        topK = topK, minSupport = minSupport, cleanText = cleanText, cleanKeys = cleanKeys,
        whiteListKeys = whiteListKeys, blackListKeys = blackListKeys,
        others = domains.tail, trackNulls = trackNulls, maxPctCardinality = maxPctCardinality
      )
    }
  }

  /**
   * Enrichment functions for URLMap Features
   *
   * @param f FeatureLike of URLMap
   */
  implicit class RichURLMapFeature(val f: FeatureLike[URLMap]) {

    /**
     * Transform URLMap feature to PickListMap by extracting domains of valid urls, converting them
     * to PickList and then vectorize the PickListMap
     *
     * @param topK              number of values to keep for each key
     * @param minSupport        min times a value must occur to be retained in pivot
     * @param cleanText         clean text after email split but before pivoting
     * @param cleanKeys         clean map keys before pivoting
     * @param whiteListKeys     keys to whitelist
     * @param blackListKeys     keys to blacklist
     * @param trackNulls        option to keep track of values that were missing
     * @param maxPctCardinality max percentage of distinct values a categorical feature can have (between 0.0 and 1.00)
     *
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      others: Array[FeatureLike[URLMap]] = Array.empty,
      maxPctCardinality: Double = OpOneHotVectorizer.MaxPctCardinality
    ): FeatureLike[OPVector] = {
      val domains: Array[FeatureLike[PickListMap]] = (f +: others).map { e =>
        val transformer =
          new UnaryLambdaTransformer[URLMap, PickListMap](
            operationName = "urlMapToPickListMap",
            transformFn = _.value
              .mapValues(v => if (v.toURL.isValid) v.toURL.domain else None)
              .collect { case (k, Some(v)) => k -> v }.toPickListMap
          )
        transformer.setInput(e).getOutput()
      }

      domains.head.vectorize(
        topK = topK, minSupport = minSupport, cleanText = cleanText, cleanKeys = cleanKeys,
        whiteListKeys = whiteListKeys, blackListKeys = blackListKeys,
        others = domains.tail, trackNulls = trackNulls, maxPctCardinality = maxPctCardinality
      )
    }
  }

  /**
   * Enrichment functions for Prediction Features
   *
   * @param f FeatureLike of URLMap
   */
  implicit class RichPredictionFeature(val f: FeatureLike[Prediction]) {

    /**
     * Takes single output feature from model of type Prediction and flattens it into 3 features
     * @return prediction, rawPrediction, probability
     */
    def tupled(): (FeatureLike[RealNN], FeatureLike[OPVector], FeatureLike[OPVector]) = {
      (f.map[RealNN](_.prediction.toRealNN),
        f.map[OPVector]{ p => Vectors.dense(p.rawPrediction).toOPVector },
        f.map[OPVector]{ p => Vectors.dense(p.probability).toOPVector }
      )
    }

    /**
     * Apply PredictionDescaler shortcut function.  Applies the inverse of the scaling function found in
     * the metadata of the the input feature: scaledFeature
     * @param scaledFeature Feature containing the metadata to reconstruct the inverse scaling function
     * @tparam I feature type of the input feature: scaledFeature
     * @tparam O Output Feature type
     * @return the scaled prediction value cast to type O.
     */
    def descale[I <: Real : TypeTag, O <: Real: TypeTag](scaledFeature: FeatureLike[I]): FeatureLike[O] = {
      new PredictionDescaler[I, O]().setInput(f, scaledFeature).getOutput()
    }
  }

}
