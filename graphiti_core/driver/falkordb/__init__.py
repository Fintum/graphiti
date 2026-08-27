"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

STOPWORDS_EN = [
    'a', 'is', 'the', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
    'into', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'their', 'then', 'there',
    'these', 'they', 'this', 'to', 'was', 'will', 'with',
]

STOPWORDS_ES = [
    'a', 'al', 'algo', 'ante', 'como', 'con', 'de', 'del', 'desde', 'donde', 'durante', 'e',
    'el', 'ella', 'ellas', 'ellos', 'en', 'entre', 'era', 'eran', 'es', 'esa', 'esas', 'ese',
    'eso', 'esos', 'esta', 'estaba', 'estaban', 'estas', 'este', 'esto', 'estos', 'fue',
    'fueron', 'ha', 'había', 'han', 'hasta', 'hay', 'la', 'las', 'le', 'les', 'lo', 'los',
    'más', 'me', 'mi', 'mis', 'muy', 'ni', 'no', 'nos', 'o', 'para', 'pero', 'por', 'que',
    'qué', 'se', 'sea', 'ser', 'si', 'sí', 'sin', 'sobre', 'son', 'su', 'sus', 'también',
    'te', 'tiene', 'tienen', 'todo', 'todos', 'tu', 'tus', 'un', 'una', 'unas', 'uno',
    'unos', 'y', 'ya', 'yo',
]

STOPWORDS = sorted(set(STOPWORDS_EN) | set(STOPWORDS_ES))
