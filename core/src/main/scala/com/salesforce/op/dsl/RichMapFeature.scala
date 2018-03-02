/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.{BinaryMap, _}
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature._

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
     * @param others        other features of the same type
     * @param topK          number of values to keep for each key
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText     clean text before pivoting
     * @param cleanKeys     clean map keys before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
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
      others: Array[FeatureLike[T]] = Array.empty
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
     * @param others        other features of the same type
     * @param topK          number of values to keep for each key
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText     clean text before pivoting
     * @param cleanKeys     clean map keys before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param typeHint      optional hint for MIME type detector
     * @param trackNulls    option to keep track of values that were missing
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
      others: Array[FeatureLike[Base64Map]] = Array.empty
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
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
    ): FeatureLike[OPVector] = {
      val hashedFeatures = new TextMapHashingVectorizer[TextMap]()
        .setInput(f +: others)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setPrependFeatureName(shouldPrependFeatureName)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(false) // Null tracking does nothing here and is done from outside the vectorizer, below
        .getOutput()

      /**
       * Note: Text is tokenized into a TextList, and then null tracking is applied. For maps, we do null
       * tracking on the original features so it's slightly different. Fortunately, tokenization for TextMaps is done
       * via the tokenize function directly, rather than with an entire stage, so things should still work here.
       */
      if (trackNulls) {
        val nullIndicators = new TextMapNullEstimator[TextMap]().setInput(f +: others).getOutput()
        new VectorsCombiner().setInput(hashedFeatures, nullIndicators).getOutput()
      }
      else hashedFeatures
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
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
    ): FeatureLike[OPVector] = {
      val hashedFeatures = new TextMapHashingVectorizer[TextAreaMap]()
        .setInput(f +: others)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setPrependFeatureName(shouldPrependFeatureName)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(false) // Null tracking does nothing here and is done from outside the vectorizer, below
        .getOutput()

      /* Note: Text is tokenized into a TextList, and then null tracking is applied. For maps, we do null
        tracking on the original features so it's slightly different. Fortunately, tokenization for TextMaps is done
        via the tokenize function directly, rather than with an entire stage, so things should still work here.
       */
      if (trackNulls) {
        val nullIndicators = new TextMapNullEstimator[TextAreaMap]().setInput(f +: others).getOutput()
        new VectorsCombiner().setInput(hashedFeatures, nullIndicators).getOutput()
      }
      else hashedFeatures
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
     * @param others        other features of the same type
     * @param topK          number of values to keep for each key
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText     clean text before pivoting
     * @param cleanKeys     clean map keys before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
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
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
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
     * Apply RealMapVectorizer on any OPMap that has double values
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
      fillWithMean: Boolean = TransmogrifierDefaults.FillWithMean,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
    ): FeatureLike[OPVector] = {
      new RealMapVectorizer[T]()
        .setInput(f +: others)
        .setFillWithMean(fillWithMean)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .getOutput()
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
     * Apply IntegralMapVectorizer on any OPMap that has long values
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
      fillWithMode: Boolean = TransmogrifierDefaults.FillWithMode,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls
    ): FeatureLike[OPVector] = {
      new IntegralMapVectorizer[T]()
        .setInput(f +: others)
        .setFillWithMode(fillWithMode)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for OPMap Features with Date values
   *
   * @param f FeatureLike
   */
  implicit class RichDateMapFeature(val f: FeatureLike[DateMap]) {

    /**
     * Apply DateMapVectorizer on any OPMap that has long values
     *
     * @param others        other features of the same type
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     * @param referenceDate reference date to subtract off before converting to vector
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[DateMap]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      referenceDate: org.joda.time.DateTime = TransmogrifierDefaults.ReferenceDate
    ): FeatureLike[OPVector] = {
      new DateMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .setReferenceDate(referenceDate)
        .getOutput()
    }
  }

  /**
   * Enrichment functions for OPMap Features with DateTime values
   *
   * @param f FeatureLike
   */
  implicit class RichDateTimeMapFeature(val f: FeatureLike[DateTimeMap]) {
    /**
     * Apply DateMapVectorizer on any OPMap that has long values
     *
     * @param others        other features of the same type
     * @param defaultValue  value to give missing keys on pivot
     * @param cleanKeys     clean text before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
     * @param referenceDate reference date to subtract off before converting to vector
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = TransmogrifierDefaults.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[DateTimeMap]] = Array.empty,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      referenceDate: org.joda.time.DateTime = TransmogrifierDefaults.ReferenceDate
    ): FeatureLike[OPVector] = {
      new DateMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .setTrackNulls(trackNulls)
        .setReferenceDate(referenceDate)
        .getOutput()
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
     * @param topK          number of values to keep for each key
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText     clean text after email split but before pivoting
     * @param cleanKeys     clean map keys before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
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
      others: Array[FeatureLike[EmailMap]] = Array.empty
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
        others = domains.tail, trackNulls = trackNulls
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
     * @param topK          number of values to keep for each key
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText     clean text after email split but before pivoting
     * @param cleanKeys     clean map keys before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     * @param trackNulls    option to keep track of values that were missing
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
      others: Array[FeatureLike[URLMap]] = Array.empty
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
        others = domains.tail, trackNulls = trackNulls
      )
    }

  }

}
