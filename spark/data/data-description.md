# Data Description

## `badges`

Badges given to a user.

 | Column name | Content description                          |
 | ----------- | -------------------------------------------- |
 | UserId      | Id of the user who received the badge        |
 | Name        | Name of the badge                            |
 | Date        | Date and time of receiving the badge by user |
 | Class       | Class of the badge                           |

## `comments`

Comments written for a post.

| Column name  | Content description                                                     |
| ------------ | ----------------------------------------------------------------------- |
| PostId       | PostId of which the comment is written for                              |
| Score        | Score of the comment (= number of upvotes – number of downvotes)        |
| Text         | Text of the comment; This filed is encoded using base64 encoding method |
| CreationDate | Date and time of the creation of the comment                            |
| UserId       | Id of the user who wrote the comment                                    |

## `posts`

The data related to posts. A post is either an answer or a question.

| Column name      | Content description                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Id               | A unique id, identifying a post                                                                                                                             |
| ParentId         | Only available for a answer; Id of the question of the answer                                                                                               |
| PostTypeId       | A number indicating the type of a post (1= a question; 2=an answer)                                                                                         |
| CreationDate     | Date and time of the creation of the post                                                                                                                   |
| Score            | Score of the post (= number of upvotes – number of downvotes)                                                                                               |
| ViewCount        | Number of times the post has been viewed or seen                                                                                                            |
| Body             | Text of the post; This filed is encoded using base64 encoding method                                                                                        |
| OwnerUserId      | Id of the user who is the owner of the post. Usually, it is the created of the post                                                                         |
| LastActivityDate | Date and time when something like an edit happened. It also may refer to the date and time when a new answer was posted, or a bounty was set for a question |
| Title            | Title of the post (only exists for a question)                                                                                                              |
| Tags             | List of tags assigned to a post (only exists for a question)                                                                                                |
| AnswerCount      | Number of answers given to a post (only valid for a question)                                                                                               |
| CommentCount     | Number of answers given to a post                                                                                                                           |
| FavoriteCount    | Number of times the post was chosen as a favourite post by users (only valid for a question)                                                                |
| CloseDate        | Date and time when a post has been closed (only exists for a question)                                                                                      |

## `users`

User data.

| Column name    | Content description                                                                   |
| -------------- | ------------------------------------------------------------------------------------- |
| Id             | A unique number indicating the ID of the user                                         |
| Reputation     | User reputation                                                                       |
| CreationDate   | Date and time when the user's account was created                                     |
| DisplayName    | Displayed name of the user                                                            |
| LastAccessDate | Date and time when the user last loaded a page                                        |
| AboutMe        | User's "about me" text                                                                |
| Views          | Number of views user's profile had (before the date and time of dumping the database) |
| UpVotes        | Sum of the total number of times user cast an upvotes                                 |
| DownVotes      | Sum of the total number of times user cast a downvotes                                |
