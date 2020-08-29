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
  case object Bulgarian extends Language("bg")
  case object Bengali extends Language("bn")
  case object Brazilian extends Language("br")
  case object Catalan extends Language("ca")
  case object Sorani extends Language("ckb")
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
  case object Sami extends Language("se")
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
  case object Tur extends Language("tur")
  case object Bel extends Language("bel")
  case object San extends Language("san")
  case object Ara extends Language("ara")
  case object Mon extends Language("mon")
  case object Tel extends Language("tel")
  case object Sin extends Language("sin")
  case object Pes extends Language("pes")
  case object Min extends Language("min")
  case object Cmn extends Language("cmn")
  case object Aze extends Language("aze")
  case object Fao extends Language("fao")
  case object Ita extends Language("ita")
  case object Ceb extends Language("ceb")
  case object Mkd extends Language("mkd")
  case object Eng extends Language("eng")
  case object Nno extends Language("nno")
  case object Lvs extends Language("lvs")
  case object Kor extends Language("kor")
  case object Som extends Language("som")
  case object Swa extends Language("swa")
  case object Hun extends Language("hun")
  case object Fra extends Language("fra")
  case object Nld extends Language("nld")
  case object Mlt extends Language("mlt")
  case object Bak extends Language("bak")
  case object Ekk extends Language("ekk")
  case object Ron extends Language("ron")
  case object Gle extends Language("gle")
  case object Hin extends Language("hin")
  case object Est extends Language("est")
  case object Tha extends Language("tha")
  case object Slk extends Language("slk")
  case object Ltz extends Language("ltz")
  case object Kan extends Language("kan")
  case object Eus extends Language("eus")
  case object Epo extends Language("epo")
  case object Bos extends Language("bos")
  case object Pol extends Language("pol")
  case object Nep extends Language("nep")
  case object Lit extends Language("lit")
  case object War extends Language("war")
  case object Srp extends Language("srp")
  case object Ces extends Language("ces")
  case object Che extends Language("che")
  case object Lav extends Language("lav")
  case object Nds extends Language("nds")
  case object Dan extends Language("dan")
  case object Mar extends Language("mar")
  case object Nan extends Language("nan")
  case object Glg extends Language("glg")
  case object Gsw extends Language("gsw")
  case object Fry extends Language("fry")
  case object Uzb extends Language("uzb")
  case object Mal extends Language("mal")
  case object Vol extends Language("vol")
  case object Fas extends Language("fas")
  case object Msa extends Language("msa")
  case object Cym extends Language("cym")
  case object Nob extends Language("nob")
  case object Ben extends Language("ben")
  case object Kaz extends Language("kaz")
  case object Heb extends Language("heb")
  case object Bre extends Language("bre")
  case object Jav extends Language("jav")
  case object Sqi extends Language("sqi")
  case object Kir extends Language("kir")
  case object Cat extends Language("cat")
  case object Oci extends Language("oci")
  case object Vie extends Language("vie")
  case object Kat extends Language("kat")
  case object Tam extends Language("tam")
  case object Tgk extends Language("tgk")
  case object Mri extends Language("mri")
  case object Slv extends Language("slv")
  case object Lat extends Language("lat")
  case object Tgl extends Language("tgl")
  case object Pan extends Language("pan")
  case object Swe extends Language("swe")
  case object Lim extends Language("lim")
  case object Tat extends Language("tat")
  case object Ell extends Language("ell")
  case object Afr extends Language("afr")
  case object Pus extends Language("pus")
  case object Isl extends Language("isl")
  case object Sun extends Language("sun")
  case object Urd extends Language("urd")
  case object Hye extends Language("hye")
  case object Hrv extends Language("hrv")
  case object Ast extends Language("ast")
  case object Rus extends Language("rus")
  case object Spa extends Language("spa")
  case object Ind extends Language("ind")
  case object Pnb extends Language("pnb")
  case object Bul extends Language("bul")
  case object Plt extends Language("plt")
  case object Deu extends Language("deu")
  case object Zul extends Language("zul")
  case object Ukr extends Language("ukr")
  case object Jpn extends Language("jpn")
  case object Por extends Language("por")
  case object Guj extends Language("guj")
  case object Fin extends Language("fin")
  case object Unknown extends Language("unknown")
}
