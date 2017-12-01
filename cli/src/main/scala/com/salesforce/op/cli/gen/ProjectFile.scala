/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import java.io.{File, FileInputStream, InputStream}
import java.nio.file.attribute.{PosixFilePermission, PosixFilePermissions}

/**
 * Represents a file that should be created in the new project.
 *
 * @param path The path of the new file to be created, relative to the root of the project
 * @param source The source of the new file, either as a stored string or streaming
 * @param perms The permissions of the file to be created
 */
case class ProjectFile(path: String, source: FileSource, perms: FilePermissions = FilePermissions()) {

  /**
   * Create a new ProjectFile by changing this project file's path.
   *
   * @param newExt The new extension
   * @return The changed [[ProjectFile]]
   */
  def withExtension(newExt: String): ProjectFile = {
    val lastDot = path.lastIndexOf(".")
    val oldBasename = path.substring(0, lastDot)
    val newPath = s"$oldBasename.$newExt"
    ProjectFile(newPath, source, perms)
  }

  /**
   * Make this project file be "replaced" from the given file. Replacement means that the given file will be
   * copied into the project directory, with the same name. This is used for copying the avro schema or data csv in the
   * template.
   *
   * @param file The file to copy from
   * @return The changed [[ProjectFile]]
   */
  def replaceFrom(file: File): ProjectFile = {
    val oldBase = path.substring(0, path.lastIndexOf('/') + 1)
    val finalPath = oldBase + file.getName
    copy(path = finalPath, source = FileSource.Streaming(new FileInputStream(file)))
  }

}

/**
 * Represents the permissions that a new file should have.
 *
 * @param perms The permissions as a Java-compatible set
 */
case class FilePermissions(perms: java.util.Set[PosixFilePermission] = FilePermissions.defaultPerms)

object FilePermissions {
  private val defaultPerms = PosixFilePermissions.fromString("rw-r--r--")
  private val execPerms = PosixFilePermissions.fromString("rwxr-xr-x")

  val default: FilePermissions = FilePermissions(defaultPerms)
  val exec: FilePermissions = FilePermissions(execPerms)
}

/**
 * Represents the "source" of a file that should be created in the new project.
 * This can be an InputStream (usually to stream straight from the CLI jar), or a String (usually from a rendered
 * template).
 */
sealed trait FileSource

object FileSource {

  /**
   * Render the [[ProjectFile]] using an in-memory string.
   */
  case class Str(source: String) extends FileSource

  /**
   * Render the [[ProjectFile]] by copying from an input stream.
   */
  case class Streaming(inputStream: InputStream) extends FileSource
}
