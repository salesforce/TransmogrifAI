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

    /** filters map by whitelisted and blacklisted keys
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
   * Enrichment functions for OPMap Features with String values
   *
   * @param f FeatureLike
   */
  implicit class RichTextMapFeature[T <: OPMap[String] : TypeTag](val f: FeatureLike[T])
    (implicit val ttiv: TypeTag[T#Value]) {

    /**
     * Apply TextMapVectorizer on any OPMap that has string values
     *
     * @param others        other features of the same type
     * @param topK          number of values to keep for each key
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText     clean text before pivoting
     * @param cleanKeys     clean map keys before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     *
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] = {
      new TextMapVectorizer[T]()
        .setInput(f +: others)
        .setTopK(topK)
        .setCleanKeys(cleanKeys)
        .setCleanText(cleanText)
        .setMinSupport(minSupport)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
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
     * @param others        other features of the same type
     * @param topK          number of values to keep for each key
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText     clean text before pivoting
     * @param cleanKeys     clean map keys before pivoting
     * @param whiteListKeys keys to whitelist
     * @param blackListKeys keys to blacklist
     *
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] = {
      new MultiPickListMapVectorizer[T]()
        .setInput(f +: others)
        .setTopK(topK)
        .setMinSupport(minSupport)
        .setCleanText(cleanText)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
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
     *
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] = {
      new RealMapVectorizer[T]()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
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
     *
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] = {
      new IntegralMapVectorizer[T]()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .getOutput()
    }

  }


  /**
   * Enrichment functions for OPMap Features with Long values
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
     *
     * @return an OPVector feature
     */
    def vectorize(
      defaultValue: Double,
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[BinaryMap]] = Array.empty
    ): FeatureLike[OPVector] = {
      new BinaryMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
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
     *
     * @return an OPVector feature
     */
    def vectorize(
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      defaultValue: Geolocation = Transmogrifier.DefaultGeolocation,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[GeolocationMap]] = Array.empty
    ): FeatureLike[OPVector] = {
      new GeolocationMapVectorizer()
        .setInput(f +: others)
        .setDefaultValue(defaultValue)
        .setCleanKeys(cleanKeys)
        .setWhiteListKeys(whiteListKeys)
        .setBlackListKeys(blackListKeys)
        .getOutput()
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
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
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
        others = domains.tail
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
     * @return an OPVector feature
     */
    def vectorize(
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      cleanKeys: Boolean = Transmogrifier.CleanKeys,
      whiteListKeys: Array[String] = Array.empty,
      blackListKeys: Array[String] = Array.empty,
      others: Array[FeatureLike[URLMap]] = Array.empty
    ): FeatureLike[OPVector] = {

      val domains: Array[FeatureLike[PickListMap]] = (f +: others).map { e =>
        val transformer = new OPMapTransformer[URL, PickList, URLMap, PickListMap](
          operationName = "urlToPickListMap",
          transformer = new UnaryLambdaTransformer[URL, PickList](
            operationName = "urlToPickList",
            transformFn = v => if (v.isValid) v.domain.toPickList else PickList.empty
          )
        )
        transformer.setInput(e).getOutput()
      }

      domains.head.vectorize(
        topK = topK, minSupport = minSupport, cleanText = cleanText, cleanKeys = cleanKeys,
        whiteListKeys = whiteListKeys, blackListKeys = blackListKeys,
        others = domains.tail
      )
    }

  }

}
