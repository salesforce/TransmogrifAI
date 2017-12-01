/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.text

import enumeratum._

/**
 * Language detector trait
 */
trait LanguageDetector extends Serializable {

  /**
   * Detect languages from a text
   *
   * @param s input text
   * @return detected languages sorted by confidence score in descending order.
   *         Confidence score is range of [0.0, 1.0], with higher values implying greater confidence.
   */
  def detectLanguages(s: String): Seq[(Language, Double)]

}

/**
 * Language enum
 * @param entryName ISO 639-1 or 639-3 language code, eg "fr" or "gsw"
 */
sealed abstract class Language(override val entryName: String) extends EnumEntry with Serializable
/**
 * Language enum with entryName is ISO 639-1 or 639-3 language code, eg "fr" or "gsw"
 */
object Language extends Enum[Language] {
  val values = findValues
  // Sorted alphabetically by entryName
  case object Afrikaans extends Language("af")
  case object Aragonese extends Language("an")
  case object Arabic extends Language("ar")
  case object Asturian extends Language("ast")
  case object Belarusian extends Language("be")
  case object Breton extends Language("br")
  case object Catalan extends Language("ca")
  case object Bulgarian extends Language("bg")
  case object Bengali extends Language("bn")
  case object Czech extends Language("cs")
  case object Welsh extends Language("cy")
  case object Danish extends Language("da")
  case object German extends Language("de")
  case object Greek extends Language("el")
  case object English extends Language("en")
  case object Spanish extends Language("es")
  case object Estonian extends Language("et")
  case object Basque extends Language("eu")
  case object Persian extends Language("fa")
  case object Finnish extends Language("fi")
  case object French extends Language("fr")
  case object Irish extends Language("ga")
  case object Galician extends Language("gl")
  case object Gujarati extends Language("gu")
  case object Hebrew extends Language("he")
  case object Hindi extends Language("hi")
  case object Croatian extends Language("hr")
  case object Haitian extends Language("ht")
  case object Hungarian extends Language("hu")
  case object Indonesian extends Language("id")
  case object Icelandic extends Language("is")
  case object Italian extends Language("it")
  case object Japanese extends Language("ja")
  case object Khmer extends Language("km")
  case object Kannada extends Language("kn")
  case object Korean extends Language("ko")
  case object Lithuanian extends Language("lt")
  case object Latvian extends Language("lv")
  case object Macedonian extends Language("mk")
  case object Malayalam extends Language("ml")
  case object Marathi extends Language("mr")
  case object Malay extends Language("ms")
  case object Maltese extends Language("mt")
  case object Nepali extends Language("ne")
  case object Dutch extends Language("nl")
  case object Norwegian extends Language("no")
  case object Occitan extends Language("oc")
  case object Punjabi extends Language("pa")
  case object Polish extends Language("pl")
  case object Portuguese extends Language("pt")
  case object Romanian extends Language("ro")
  case object Russian extends Language("ru")
  case object Slovak extends Language("sk")
  case object Slovene extends Language("sl")
  case object Somali extends Language("so")
  case object Albanian extends Language("sq")
  case object Serbian extends Language("sr")
  case object Swedish extends Language("sv")
  case object Swahili extends Language("sw")
  case object Tamil extends Language("ta")
  case object Telugu extends Language("te")
  case object Thai extends Language("th")
  case object Tagalog extends Language("tl")
  case object Turkish extends Language("tr")
  case object Ukrainian extends Language("uk")
  case object Urdu extends Language("ur")
  case object Vietnamese extends Language("vi")
  case object Walloon extends Language("wa")
  case object Yiddish extends Language("yi")
  case object SimplifiedChinese extends Language("zh-cn")
  case object TraditionalChinese extends Language("zh-tw")
  case object Unknown extends Language("unknown")
}
