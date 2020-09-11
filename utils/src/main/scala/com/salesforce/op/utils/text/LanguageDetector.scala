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
sealed abstract class Language(
  override val entryName: String
) extends EnumEntry with Serializable {
  val langCodeSplit = entryName.split(":")
  val lang3 = {
    if (entryName.contains(":")) langCodeSplit(1)
    else if (entryName.length == 3) entryName
    else ""
  }

  val lang2 = {
    if (entryName.contains(":")) langCodeSplit(0)
    else if (entryName.length == 2) entryName
    else ""
  }
}
/**
 * Language enum with entryName is ISO 639-1 or 639-3 language code, eg "fr" or "gsw"
 */
object Language extends Enum[Language] {
  val values = findValues
  val lang2Lookup = values.collect { case v if v.lang2.length > 0 => v.lang2 -> v }.toMap
  val lang3Lookup = values.collect { case v if v.lang3.length > 0 => v.lang3 -> v }.toMap

  def fromString(str: String): Language = {
    lang2Lookup.getOrElse(str, lang3Lookup.getOrElse(str, Unknown))
  }

  // Sorted alphabetically by entryName
  case object Afrikaans extends Language("af:afr")
  case object Albanian extends Language("sq")
  case object Arabic extends Language("ar:ara")
  case object Aragonese extends Language("an:arg")
  case object Asturian extends Language("ast")
  case object Basque extends Language("eu:eus")
  case object Belarusian extends Language("be:bel")
  case object Bengali extends Language("bn:ben")
  case object Brazilian extends Language("br")
  case object Breton extends Language("br:bre")
  case object Bulgarian extends Language("bg:bul")
  case object Catalan extends Language("ca")
  case object Croatian extends Language("hr")
  case object Czech extends Language("cs")
  case object Danish extends Language("da")
  case object Dutch extends Language("nl")
  case object English extends Language("en:eng")
  case object Estonian extends Language("et")
  case object Finnish extends Language("fi:fin")
  case object French extends Language("fr")
  case object Galician extends Language("gl")
  case object German extends Language("de:deu")
  case object Greek extends Language("el")
  case object Gujarati extends Language("gu:guj")
  case object Haitian extends Language("ht")
  case object Hebrew extends Language("he")
  case object Hindi extends Language("hi")
  case object Hungarian extends Language("hu")
  case object Icelandic extends Language("is")
  case object Indonesian extends Language("id")
  case object Irish extends Language("ga")
  case object Italian extends Language("it:ita")
  case object IranianPersian extends Language("pes")
  case object Japanese extends Language("ja:jpn")
  case object Kannada extends Language("kn")
  case object Khmer extends Language("km")
  case object Korean extends Language("ko")
  case object Latvian extends Language("lv")
  case object Lithuanian extends Language("lt")
  case object Macedonian extends Language("mk")
  case object Malay extends Language("ms")
  case object Malayalam extends Language("ml")
  case object Maltese extends Language("mt")
  case object Marathi extends Language("mr")
  case object Minangkabau extends Language("min")
  case object Mongolian extends Language("mn:mon")
  case object Nepali extends Language("ne")
  case object Norwegian extends Language("no")
  case object Occitan extends Language("oc")
  case object Persian extends Language("fa:fas")
  case object Polish extends Language("pl")
  case object Portuguese extends Language("pt")
  case object Punjabi extends Language("pa")
  case object Romanian extends Language("ro")
  case object Russian extends Language("ru")
  case object Sami extends Language("se")
  case object Sanskrit extends Language("sa:san")
  case object Serbian extends Language("sr")
  case object SimplifiedChinese extends Language("zh-cn:cmn")
  case object Sinhalese extends Language("si:sin")
  case object Slovak extends Language("sk")
  case object Slovene extends Language("sl")
  case object Somali extends Language("so")
  case object Sorani extends Language("ckb")
  case object Spanish extends Language("es")
  case object Sundanese extends Language("su:sun")
  case object Swahili extends Language("sw")
  case object Swedish extends Language("sv")
  case object Tagalog extends Language("tl")
  case object Tamil extends Language("ta")
  case object Telugu extends Language("te:tel")
  case object Thai extends Language("th")
  case object TraditionalChinese extends Language("zh-tw")
  case object Turkish extends Language("tr:tur")
  case object Ukrainian extends Language("uk")
  case object Urdu extends Language("ur")
  case object Vietnamese extends Language("vi")
  case object Walloon extends Language("wa")
  case object Welsh extends Language("cy")
  case object Yiddish extends Language("yi")
  case object Min extends Language("min")
  case object Cmn extends Language("cmn")
  case object Aze extends Language("aze")
  case object Fao extends Language("fao")
  case object Ita extends Language("ita")
  case object Ceb extends Language("ceb")
  case object Mkd extends Language("mkd")
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
  case object Msa extends Language("msa")
  case object Cym extends Language("cym")
  case object Nob extends Language("nob")
  case object Kaz extends Language("kaz")
  case object Heb extends Language("heb")
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
  case object Urd extends Language("urd")
  case object Hye extends Language("hye")
  case object Hrv extends Language("hrv")
  case object Rus extends Language("rus")
  case object Spa extends Language("spa")
  case object Ind extends Language("ind")
  case object Pnb extends Language("pnb")
  case object Plt extends Language("plt")
  case object Zul extends Language("zul")
  case object Ukr extends Language("ukr")
  case object Por extends Language("por")
  case object Unknown extends Language("unknown")
}
