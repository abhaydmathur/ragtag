# Resource URL: https://developers.facebook.com/docs/graph-api/reference/v18.0/request
This document refers to an outdated version of Graph API. Please [use the latest version.](https://developers.facebook.com/docs/graph-api/reference/v19.0/request)

Request `/{request-id}`
=======================

An individual game request received by someone, sent by an app or by another person.

### Related Guides

* [Game Requests](https://developers.facebook.com/docs/games/requests/)
    
* [`/{user-id}/apprequests`](https://developers.facebook.com/docs/graph-api/reference/user/apprequests/)
    

Reading
-------

HTTPPHP SDKJavaScript SDKAndroid SDKiOS SDK[Graph API Explorer](https://developers.facebook.com/tools/explorer/?method=GET&path=%7Brequest-id%7D&version=v19.0)

    GET /v19.0/{request-id} HTTP/1.1
    Host: graph.facebook.com

    /* PHP SDK v5.0.0 */
    /* make the API call */
    try {
      // Returns a `Facebook\FacebookResponse` object
      $response = $fb->get(
        '/{request-id}',
        '{access-token}'
      );
    } catch(Facebook\Exceptions\FacebookResponseException $e) {
      echo 'Graph returned an error: ' . $e->getMessage();
      exit;
    } catch(Facebook\Exceptions\FacebookSDKException $e) {
      echo 'Facebook SDK returned an error: ' . $e->getMessage();
      exit;
    }
    $graphNode = $response->getGraphNode();
    /* handle the result */

    /* make the API call */
    FB.api(
        "/{request-id}",
        function (response) {
          if (response && !response.error) {
            /* handle the result */
          }
        }
    );

    /* make the API call */
    new GraphRequest(
        AccessToken.getCurrentAccessToken(),
        "/{request-id}",
        null,
        HttpMethod.GET,
        new GraphRequest.Callback() {
            public void onCompleted(GraphResponse response) {
                /* handle the result */
            }
        }
    ).executeAsync();

    /* make the API call */
    FBSDKGraphRequest *request = [[FBSDKGraphRequest alloc]
                                   initWithGraphPath:@"/{request-id}"
                                          parameters:params
                                          HTTPMethod:@"GET"];
    [request startWithCompletionHandler:^(FBSDKGraphRequestConnection *connection,
                                          id result,
                                          NSError *error) {
        // Handle the result
    }];

### Permissions

* A user access token is required if you are requesting using only the Request object ID, and want to know about the recipient of the request. The request must have been sent to the person whose access token you are using.
    
* An app access token can be used when you are requesting using the concatenated Request object ID and user ID string, or when you are only using the request object ID, but do not need to know recipient info. See the [Requests docs](https://developers.facebook.com/docs/howtos/requests/) for more info on this ID.
    

### Fields

| Name | Description | Type |
| --- | --- | --- |
| `id` | The request object ID. | `string` |
| `application` | App associated with the request. | [`App`](https://developers.facebook.com/docs/graph-api/reference/app/) |
| `to` | The recipient of the request. | [`User`](https://developers.facebook.com/docs/graph-api/reference/user/) |
| `from` | The sender associated with the request. This is only included for [user to user requests](https://developers.facebook.com/docs/howtos/requests#user_to_user). | [`User`](https://developers.facebook.com/docs/graph-api/reference/user/) |
| `message` | A string describing the request. | `string` |
| `created_time` | Timestamp when the request was created. | `datetime` |

Publishing
----------

You can't publish using this endpoint.

Requests are published via the [Game Request Dialog](https://developers.facebook.com/docs/games/requests). If your app is a **Game** you can publish app requests using the [`/{user-id}/apprequests` edge](https://developers.facebook.com/docs/graph-api/reference/user/apprequests/).

Deleting
--------

HTTPPHP SDKJavaScript SDKAndroid SDKiOS SDK

    DELETE /v19.0/{request-id} HTTP/1.1
    Host: graph.facebook.com

    /* PHP SDK v5.0.0 */
    /* make the API call */
    try {
      // Returns a `Facebook\FacebookResponse` object
      $response = $fb->delete(
        '/{request-id}',
        array (),
        '{access-token}'
      );
    } catch(Facebook\Exceptions\FacebookResponseException $e) {
      echo 'Graph returned an error: ' . $e->getMessage();
      exit;
    } catch(Facebook\Exceptions\FacebookSDKException $e) {
      echo 'Facebook SDK returned an error: ' . $e->getMessage();
      exit;
    }
    $graphNode = $response->getGraphNode();
    /* handle the result */

    /* make the API call */
    FB.api(
        "/{request-id}",
        "DELETE",
        function (response) {
          if (response && !response.error) {
            /* handle the result */
          }
        }
    );

    /* make the API call */
    new GraphRequest(
        AccessToken.getCurrentAccessToken(),
        "/{request-id}",
        null,
        HttpMethod.DELETE,
        new GraphRequest.Callback() {
            public void onCompleted(GraphResponse response) {
                /* handle the result */
            }
        }
    ).executeAsync();

    /* make the API call */
    FBSDKGraphRequest *request = [[FBSDKGraphRequest alloc]
                                   initWithGraphPath:@"/{request-id}"
                                          parameters:params
                                          HTTPMethod:@"DELETE"];
    [request startWithCompletionHandler:^(FBSDKGraphRequestConnection *connection,
                                          id result,
                                          NSError *error) {
        // Handle the result
    }];

### Permissions

* A user access token is required if you are using only the Request object ID. The request must also have been sent to the person whose access token you are using.
    
* An app access token can be used when you are using the concatenated Request object ID and user ID string.
    

### Fields

No fields are needed to delete.

### Response

If successful:

{
  "success": true
}

Otherwise a relevant error message will be returned.

Updating
--------

You can't update using this endpoint.

![](https://www.facebook.com/tr?id=675141479195042&ev=PageView&noscript=1)

![](https://www.facebook.com/tr?id=574561515946252&ev=PageView&noscript=1)

![](https://www.facebook.com/tr?id=1754628768090156&ev=PageView&noscript=1)