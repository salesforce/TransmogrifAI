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

import com.google.i18n.phonenumbers.{PhoneNumberUtil, Phonenumber}
import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.stats.JaccardSim
import org.apache.spark.ml.param.{BooleanParam, Param, Params, StringArrayParam}

import scala.util.Try

trait PhoneParams extends Params {

  final val strictValidation = new BooleanParam(
    parent = this, name = "strictValidation",
    doc = "If true will evaulate phone number as presented. Otherwise will try to search for a valid substring"
  )


  /**
   * If set to true, phone number will be compared strictly checked. If set to
   * false, an input, that is too long, will be considered valid if a substring can be found
   * to be valid
   *
   * @param flag
   * @return
   */
  def setStrictness(flag: Boolean): this.type = {
    set(strictValidation, flag)
  }


  final val defaultRegion = new Param[String](
    parent = this, name = "defaultLocale",
    doc = "Default country code to use if valid code cannot be determined"
  )

  /**
   * Sets the default country code to use when a valid code cannot be
   * determined
   *
   * @param cc Country code to check phone number validity against by default
   * @return
   */
  def setDefaultRegion(cc: String): this.type = {
    set(defaultRegion, cc)
  }

  setDefault(
    strictValidation -> PhoneNumberParser.StrictValidation,
    defaultRegion -> PhoneNumberParser.DefaultRegion
  )

}


trait PhoneCountryParams extends PhoneParams {

  final val regionCodes = new StringArrayParam(
    parent = this, name = "regionCodes", doc = "List of valid region codes",
    isValid = (cc: Array[String]) => {
      val supportRegions = PhoneNumberParser.phoneUtil.getSupportedRegions
      cc.forall(supportRegions.contains)
    }
  )

  final val countryNames = new StringArrayParam(
    parent = this, name = "countryNames", doc = "List of valid country names"
  )

  setDefault(
    regionCodes -> PhoneNumberParser.DefaultCountryCodes.keys.map(_.toUpperCase).toArray,
    countryNames -> PhoneNumberParser.DefaultCountryCodes.values.map(_.toUpperCase).toArray
  )

  /**
   * Initialize with country name and their codes. The strings are upper cased when set.
   *
   * By default this transformer will validate against the regional codes provided by Google's
   * PhoneNumber library.
   *
   * If the user would like to use country names as the second argument to this transformer,
   * a mapping between the names and their code must be provided here.
   *
   * @param countryCodes country Code -> country Name eg "US" -> "United States of America"
   * @return reference to self
   */
  def setCodesAndCountries(countryCodes: Map[String, String]): this.type = {
    set(regionCodes, countryCodes.keys.map(_.toUpperCase).toArray)
    set(countryNames, countryCodes.values.map(_.toUpperCase).toArray)
  }

}


/**
 * Determine whether a phone number is valid given the country's regional code.
 * By default the regional code will be checked against those provided in Google's
 * PhoneNumber library. If the input regional code is not found, the default locale will
 * be used for validation.
 *
 * If the User provided a Country name to code mapping, the phone number can only be
 * validated against the input mapping. This transformer will first match on regional code,
 * failing that, it will select the country with the closest Q-Distance.
 *
 * All phone numbers with less than 2 characters will be categorized as invalid
 *
 * All phone numbers that starts with "+" will be evaluated with international formatting
 *
 * Returns stripped number if number is valid. And None other wise.
 */
class ParsePhoneNumber(uid: String = UID[ParsePhoneNumber])
  extends BinaryTransformer[Phone, Text, Phone](
    operationName = "parsePhone",
    uid = uid
  ) with PhoneCountryParams {

  override def transformFn: (Phone, Text) => Phone = (phoneNumber: Phone, regionCode: Text) => {
    val code = PhoneNumberParser.validCountryCode(
      phoneNumber = phoneNumber,
      regionCode = regionCode,
      defaultRegionCode = $(defaultRegion),
      regionCodes = $(regionCodes),
      countryNames = $(countryNames)
    )
    PhoneNumberParser.parse(phoneNumber, code, $(strictValidation))
  }
}


/**
 * Transformer to determine if a phone numbers is valid when no country code is available.
 * The default locale will be used for validation.
 * All phone numbers with less than 2 characters will be categorized as invalid
 * All phone numbers that starts with "+" will be evaluate with international formatting
 *
 * Returns stripped number if number is valid. And None other wise.
 */
class ParsePhoneDefaultCountry(uid: String = UID[ParsePhoneDefaultCountry])
  extends UnaryTransformer[Phone, Phone](
    operationName = "parsePhoneNoCC",
    uid = uid
  ) with PhoneParams {

  override def transformFn: (Phone) => Phone = (phoneNumber: Phone) => {
    PhoneNumberParser.parse(phoneNumber, $(defaultRegion), $(strictValidation))
  }
}


/**
 * Determine whether a phone number is valid given the country's regional code.
 * By default the regional code will be checked against those provided in Google's
 * PhoneNumber library. If the input regional code is not found, the default locale will
 * be used for validation.
 *
 * If the User provided a Country name to code mapping, the phone number can only be
 * validated against the input mapping. This transformer will first match on regional code,
 * failing that, it will select the country with the closest Q-Distance.
 *
 * All phone numbers with less than 2 characters will be categorized as invalid
 *
 * All phone numbers that starts with "+" will be evaluated with international formatting
 *
 * Returns binary feature true if phone is valid false if invalid and none if phone number is none
 */
class IsValidPhoneNumber(uid: String = UID[IsValidPhoneNumber])
  extends BinaryTransformer[Phone, Text, Binary](
    operationName = "validatePhone",
    uid = uid
  ) with PhoneCountryParams {

  override def transformFn: (Phone, Text) => Binary = (phoneNumber: Phone, regionCode: Text) => {
    val code = PhoneNumberParser.validCountryCode(
      phoneNumber = phoneNumber,
      regionCode = regionCode,
      defaultRegionCode = $(defaultRegion),
      regionCodes = $(regionCodes),
      countryNames = $(countryNames)
    )
    PhoneNumberParser.validate(phoneNumber, code, $(strictValidation))
  }
}


/**
 * Transformer to determine if a phone numbers is valid when no country code is available.
 * The default locale will be used for validation.
 * All phone numbers with less than 2 characters will be categorized as invalid
 * All phone numbers that starts with "+" will be evaluated with international formatting
 *
 * Returns binary feature true if phone is valid false if invalid and none if phone number is none
 */
class IsValidPhoneDefaultCountry(uid: String = UID[IsValidPhoneDefaultCountry])
  extends UnaryTransformer[Phone, Binary](operationName = "validatePhoneNoCC", uid = uid) with PhoneParams {

  override def transformFn: (Phone) => Binary = (phoneNumber: Phone) => {
    PhoneNumberParser.validate(phoneNumber, $(defaultRegion), $(strictValidation))
  }
}

/**
 * Transformer to determine if a map of phone numbers is valid when no country code is available.
 * The default locale will be used for validation.
 * All phone numbers with less than 2 characters will be categorized as invalid
 * All phone numbers that starts with "+" will be evaluated with international formatting
 *
 * Returns binary map feature true if phone is valid false if invalid and none if phone number is none
 */
class IsValidPhoneMapDefaultCountry(uid: String = UID[IsValidPhoneMapDefaultCountry])
  extends UnaryTransformer[PhoneMap, BinaryMap](operationName = "validatePhoneMapNoCC", uid = uid) with PhoneParams {

  override def transformFn: (PhoneMap) => BinaryMap = (phoneNumberMap: PhoneMap) => {
    val region = $(defaultRegion)
    val isStrict = $(strictValidation)

    phoneNumberMap.value
      .mapValues(p => PhoneNumberParser.validate(p.toPhone, region, isStrict))
      .collect{ case(k, v) if !v.isEmpty => k -> v.value.get }.toBinaryMap
  }
}

case object PhoneNumberParser {
  val DefaultRegion = "US"
  val StrictValidation = false
  val InternationalCode = "ZZ" // google PhoneNumber parsing convention: Region-code for the unknown region.

  private[op] def phoneUtil = PhoneNumberUtil.getInstance()

  /**
   * trims phone numbers string and removes all non-numeric except "+" symbols
   *
   * @param pn phone number as string
   * @return
   */
  private[op] def cleanNumber(pn: String): String = pn.trim.replaceAll("[^+\\d]", "")


  private def parsePhoneNumber(
    phoneNumber: String, countryCode: String, strictValidation: Boolean
  ): Try[Phonenumber.PhoneNumber] = Try {
    val cleanedPhone = cleanNumber(phoneNumber)
    val number = phoneUtil.parse(cleanedPhone, countryCode.toUpperCase)
    if (!strictValidation) phoneUtil.truncateTooLongNumber(number)
    number
  }

  private def validateAgainstCountryCode(
    phoneNumber: String, countryCode: String, strictValidation: Boolean
  ): Try[Boolean] = parsePhoneNumber(phoneNumber, countryCode, strictValidation).map(phoneUtil.isValidNumber)

  private def isInternationalFormat(pn: String): Boolean = pn.startsWith("+")

  private[op] def validCountryCode
  (
    phoneNumber: Phone,
    regionCode: Text,
    defaultRegionCode: String,
    regionCodes: Array[String],
    countryNames: Array[String]
  ): String = {
    phoneNumber.value -> regionCode.v.map(_.toUpperCase()) match {
      case (Some(phone), _) if isInternationalFormat(phone) => InternationalCode
      case (_, Some(rc)) if regionCodes.contains(rc) => rc
      case (_, Some(rc)) if phoneUtil.getSupportedRegions.contains(rc) => rc
      case (_, Some(rc)) if regionCodes.nonEmpty =>
        val rcBi = rc.trim.sliding(2).toSet
        regionCodes.zip(countryNames).flatMap {
          case (regCode, country) => country.split(",").map{ // Can have multiple versions of country name
            c => regCode -> JaccardSim(rcBi, c.trim.sliding(2).toSet)
          }}.maxBy(_._2)._1
      case _ => defaultRegionCode
    }
  }

  private[op] def validate(phoneNumber: Phone, regionCode: String, strictValidation: Boolean): Binary = new Binary(
    phoneNumber.v.flatMap {
      case pn if pn.length() < 2 => None
      case pn => validateAgainstCountryCode(pn, regionCode, strictValidation).toOption
    }
  )

  private[op] def parse(phoneNumber: Phone, regionCode: String, strictValidation: Boolean): Phone = new Phone(
    phoneNumber.v.flatMap {
      case pn if pn.length() < 2 => None
      case pn =>
        val parsed = parsePhoneNumber(pn, regionCode, strictValidation)
        val validOpt = parsed.toOption.filter(phoneUtil.isValidNumber)
        validOpt.map(p => s"+${p.getCountryCode}${p.getNationalNumber}${p.getExtension}")
    }
  )

  // scalastyle:off
  val DefaultCountryCodes = Map(
    "US" -> "USA, United States of America",
    "CA" -> "Canada",
    "DO" -> "Dominican Republic",
    "PR" -> "Puerto Rico",
    "BS" -> "Bahamas",
    "BB" -> "Barbados",
    "AI" -> "Anguilla",
    "AG" -> "Antigua & Barbuda",
    "VG" -> "British Virgin Islands",
    "VI" -> "US Virgin Islands",
    "KY" -> "Cayman Islands",
    "BM" -> "Bermuda",
    "GD" -> "Grenada",
    "TC" -> "Turks & Caicos",
    "MS" -> "Montserrat",
    "MP" -> "Northern Mariana Islands",
    "GU" -> "Guam",
    "AS" -> "American Samoa",
    "SX" -> "Sint Maarten",
    "LC" -> "Saint Lucia",
    "DM" -> "Dominica",
    "VC" -> "Saint Vincent & the Grenadines",
    "TT" -> "Trinidad & Tobago",
    "KN" -> "Saint Kitts & Nevis",
    "JM" -> "Jamaica",
    "EG" -> "Eygyt, مصر",
    "SS" -> "South Sudan",
    "MA" -> "Morocco",
    "EH" -> "Western Sahara, SADR",
    "DZ" -> "Algeria",
    "TN" -> "Tunisia, تونس",
    "LY" -> "Libya",
    "GM" -> "Gambia",
    "SN" -> "Senegal",
    "MR" -> "Mauritania",
    "ML" -> "Mali",
    "GN" -> "Guinea",
    "CI" -> "Cote d'Ivoire, Ivory Coast",
    "BF" -> "Burkina Faso",
    "NE" -> "Niger",
    "TG" -> "Togo",
    "BJ" -> "Benin",
    "MU" -> "Mauritius",
    "LR" -> "Liberia",
    "SL" -> "Sierra Leone",
    "GH" -> "Ghana",
    "NG" -> "Nigeria",
    "TD" -> "Chad",
    "CF" -> "Central African Republic",
    "CM" -> "Cameroon",
    "CV" -> "Cape Verde",
    "ST" -> "Sao Tome and Principe",
    "GQ" -> "Equatorial Guinea",
    "GA" -> "Gabon",
    "CG" -> "Congo, Brazzaville",
    "CD" -> "Democratic Republic of Congo",
    "AO" -> "Angola",
    "GW" -> "Guinea-Bissau",
    "IO" -> "Diego Garcia",
    "AC" -> "Ascension",
    "SC" -> "Seychelles",
    "SD" -> "Sudan",
    "RW" -> "Rwanda",
    "ET" -> "Ethiopia",
    "SO" -> "Somalia, Somaliland, Puntland",
    "DJ" -> "Djibouti",
    "KE" -> "Kenya",
    "TZ" -> "Tanzania",
    "UG" -> "Uganda",
    "BI" -> "Burundi",
    "MZ" -> "Mozambique",
    "ZM" -> "Zambia",
    "MG" -> "Madagascar",
    "RE" -> "Réunion",
    "ZW" -> "Zimbabwe",
    "NA" -> "Namibia",
    "MW" -> "Malawi",
    "LS" -> "Lesotho",
    "BW" -> "Botswana",
    "SZ" -> "Swaziland",
    "KM" -> "Comoros",
    "ZA" -> "South Africa",
    "SH" -> "Saint Helena, Tristan da Cunha",
    "ER" -> "Eritrea",
    "AW" -> "Aruba",
    "FO" -> "Faroe Islands",
    "GL" -> "Greenland",
    "GR" -> "Greece",
    "NL" -> "Netherlands",
    "BE" -> "Belgium",
    "FR" -> "France",
    "ES" -> "Spain, España",
    "GI" -> "Gibraltar",
    "PT" -> "Portugal",
    "LU" -> "Luxembourg",
    "IE" -> "Ireland",
    "IS" -> "Iceland",
    "AL" -> "Albania",
    "MT" -> "Malta",
    "CY" -> "Cyprus",
    "FI" -> "Finland",
    "AX" -> "Åland",
    "BG" -> "Bulgaria, БГ",
    "HU" -> "Hungary",
    "LT" -> "Lithuania",
    "LV" -> "Latvia",
    "EE" -> "Estonia",
    "MD" -> "Moldova, Transnistria",
    "AM" -> "Armenia",
    "BY" -> "Belarus",
    "AD" -> "Andorra",
    "MC" -> "Monaco",
    "SM" -> "San Marino",
    "UA" -> "Ukraine, Украина, УКР",
    "RS" -> "Serbia, Kosovo, Vojvodina",
    "ME" -> "Montenegro",
    "HR" -> "Croatia",
    "SI" -> "Slovenia",
    "BA" -> "Bosnia and Herzegovina",
    "MK" -> "Republic of Macedonia",
    "IT" -> "Italy",
    "VA" -> "Vatican City",
    "RO" -> "Romania",
    "CH" -> "Switzerland",
    "CZ" -> "Czech Republic",
    "SK" -> "Slovakia",
    "LI" -> "Liechtenstein",
    "AT" -> "Austria",
    "GB" -> "United Kingdom, Great Britain, England, Scotland, Wales, UK",
    "GB" -> "Northern Ireland",
    "GG" -> "Guernsey",
    "IM" -> "Isle of Man",
    "JE" -> "Jersey",
    "DK" -> "Denmark",
    "SE" -> "Sweden",
    "NO" -> "Norway",
    "PL" -> "Poland",
    "DE" -> "Germany",
    "FK" -> "Falkland Islands",
    "BZ" -> "Belize",
    "GT" -> "Guatemala",
    "SV" -> "El Salvador",
    "HN" -> "Honduras",
    "NI" -> "Nicaragua",
    "CR" -> "Costa Rica",
    "PA" -> "Panama",
    "FR" -> "St Pierre & Miquelon, PM",
    "HT" -> "Haiti",
    "PE" -> "Peru",
    "MX" -> "Mexico",
    "CU" -> "Cuba",
    "AR" -> "Argentina",
    "BR" -> "Brazil",
    "CL" -> "Chile",
    "CO" -> "Colombia",
    "VE" -> "Venezuela",
    "GP" -> "Guadeloupe",
    "BO" -> "Bolivia",
    "GY" -> "Guyana",
    "EC" -> "Ecuador",
    "GF" -> "Guiana",
    "PY" -> "Paraguay",
    "MQ" -> "Martinique",
    "SR" -> "Suriname",
    "UY" -> "Uruguay",
    "CW" -> "Curaçao",
    "MY" -> "Malaysia",
    "AU" -> "Australia",
    "CC" -> "Cocos Keeling Islands",
    "CX" -> "Christmas Island",
    "ID" -> "Indonesia",
    "PH" -> "Philippines",
    "NZ" -> "New Zealand",
    "SG" -> "Singapore",
    "TH" -> "Thailand, ไทย",
    "TL" -> "Timor Leste, East Timor",
    "NF" -> "Norfolk Island",
    "BN" -> "Brunei Darussalam",
    "NR" -> "Nauru",
    "PG" -> "Papua New Guinea",
    "TO" -> "Tonga",
    "SB" -> "Solomon Islands",
    "VU" -> "Vanuatu",
    "FJ" -> "Fiji",
    "PW" -> "Palau",
    "WF" -> "Wallis and Futuna",
    "CK" -> "Cook Islands",
    "NU" -> "Niue",
    "WS" -> "Samoa",
    "KI" -> "Kiribati",
    "NC" -> "New Caledonia",
    "TV" -> "Tuvalu",
    "PF" -> "French Polynesia",
    "TK" -> "Tokelau",
    "FM" -> "F.S. Micronesia",
    "MH" -> "Marshall Islands",
    "RU" -> "Russia, Russian Federation, Россия",
    "JP" -> "Japan, 日本",
    "KR" -> "Republic of Korea, South Korea",
    "VN" -> "Vietnam",
    "KP" -> "DPR Korea, North Korea",
    "HK" -> "Hong Kong, 香港",
    "MO" -> "Macau, Macao, 澳門",
    "KH" -> "Cambodia",
    "LA" -> "Laos",
    "CN" -> "China, 中国, 中國",
    "BD" -> "Bangladesh",
    "TW" -> "Taiwan, 台灣, 台湾",
    "TR" -> "Turkey",
    "TR" -> "Northern Cyprus",
    "IN" -> "India",
    "PK" -> "Pakistan",
    "AF" -> "Afghanistan",
    "LK" -> "Sri Lanka",
    "MM" -> "Burma, Myanmar",
    "MV" -> "Maldives",
    "LB" -> "Lebanon",
    "JO" -> "Jordan",
    "SY" -> "Syria",
    "IQ" -> "Iraq",
    "KW" -> "Kuwait",
    "SA" -> "Saudi Arabia, السعودية",
    "YE" -> "Yemen",
    "PS" -> "Palestine, فلسطين",
    "AE" -> "United Arab Emirates, امارات",
    "IL" -> "Israel",
    "BH" -> "Bahrain",
    "QA" -> "Qatar, قطر",
    "BT" -> "Bhutan",
    "MN" -> "Mongolia",
    "NP" -> "Nepal",
    "IR" -> "Iran, ایران",
    "TJ" -> "Tajikistan",
    "TM" -> "Turkmenistan",
    "AZ" -> "Azerbaijan, Nagorno Karabakh",
    "GE" -> "Republic of Georgia, Abkhazia, South Ossetia",
    "KG" -> "Kyrgyzstan",
    "UZ" -> "Uzbekistan"
  ) // TODO BL, BQ, SJ, TA, YT, KZ, MF, OM, PM
  // scalastyle:on
}
