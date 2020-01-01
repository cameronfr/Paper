// cfz, September 2018.
// TODO: novelty, 200ms wait before search to reduce lag, aesthetics, cleaning up word results.

const API_URL = "https://asia-northeast1-traininggpu.cloudfunctions.net/GutenBert"
const API_HELPER_URL = "https://asia-northeast1-traininggpu.cloudfunctions.net/GutenBertHelper"

const VECTOR_URL = "https://storage.googleapis.com/mlstorage-cloud/Data/glove.6B.50d.txt.zip"

//fetch doesn't have progress, file too big to not have indicator ==> xhr ugliness.
function fetchWordVectors(onProgress) {
  return new Promise((resolve, reject) => {
    localforage.getItem('vectorZip').then(zip => {
      if (zip == null) {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", VECTOR_URL, true);
        xhr.responseType = "arraybuffer";
        xhr.onerror = (e) => {
          console.log("✗ Error fetching vector file:\n" + xhr.statusText)
          reject(xhr.statusText)
        }
        xhr.onload = (e) => {
          let zip = xhr.response
          localforage.setItem('vectorZip', zip).then(() => {
            console.log("✓ Downloaded and stored vector file")
            resolve(zip)
          })
        }
        xhr.onprogress = (e) => {
          onProgress(e.loaded / e.total)
        }
        xhr.send();
      }
      else {
        console.log("✓ Found stored vector file")
        resolve(zip)
      }
    })
  })
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function loadWordVectors(zipArrayBuffer, onProgress) {
  var seconds = 0
  var unzipProgress = setInterval(() => {
    seconds = seconds < 50 ? seconds + 1 : seconds
    onProgress(seconds / 100.0) // simulate ZIP loading progress
  }, 140)
  let text = await JSZip.loadAsync(zipArrayBuffer).then(res => res.files["glove.6B.50d.txt"].async("string"))
    .catch(err => console.log("✗ Error unzipping vector file:\n" + err))
  clearInterval(unzipProgress)

  let lines = text.split("\n")
  lines = lines.slice(0, -1)

  let wordDict = {fromIdx: {}, fromWord: {}}
  let wordVectors = tf.buffer([lines.length, 50], "float32")

  for (var idx in lines) {
    let line = lines[idx]
    let sep = line.indexOf(" ")
    let word = line.substring(0, sep)
    let vector = line.substring(sep+1).split(' ')
    vector = vector.map(x => parseFloat(x))

    wordDict.fromIdx[parseInt(idx)] = word
    wordDict.fromWord[word] = idx

    let magnitude = Math.sqrt(vector.reduce((acc, x) => acc + Math.pow(x, 2), 0.0))

    for (var jdx in vector) {
      let feat = vector[jdx] / magnitude
      wordVectors.set(feat, parseInt(idx), parseInt(jdx))
    }

    if (idx % Math.floor(lines.length / 100) == 0) {
      onProgress(0.5*(parseInt(idx)/lines.length)+0.5)
      await sleep(1) //react needs a frame to render or something like that
    }
  }

  console.log("✓ Loaded word vector matrix of shape " + wordVectors.shape)
  wordVectors = wordVectors.toTensor()
  return [wordVectors, wordDict]

}

function getHighestKIndices(array, k) {
  k = k + 1
  let highestValues = Array(k)
  for (var i=0; i<array.length; i++) {
    for (var j=0; j<highestValues.length; j++) {
      if (highestValues[j] == undefined) {
        highestValues[j] = {val: array[i], index: i}
        break
      }
      else if (array[i] > highestValues[j].val) {
        highestValues[j].val = array[i]
        highestValues[j].index = i
        break
      }
    }
  }
  highestValues.sort((a, b) => b.val - a.val)
  highestValues = highestValues.slice(1).map(x => x.index)
  return highestValues
}

var fetchFromServer = (url, body) => {
	return fetch(url, {
		method: "POST",
		mode: "cors",
		headers: {"Content-Type": "application/json"},
		body: JSON.stringify(body)
	}).then(res => res.json())
}

var exampleQueries = [
	"cooking a stew",
	"cooking a stew for the family",
	"trees were dancing in the wind",
	"\"I never want to see you again\"",
	"looking for inspiration for his next project",
	"walking through the tall trees",
	"they were harvesting apples for cider",
	"the fire crackled as we sat around it telling stories",
	"a fire engulfing the house",
	"hi there",
	"are we alone in the universe?"
]

class App extends React.Component {

  appConfig = new blockstack.AppConfig(['store_write'])

  constructor(props) {
    super(props)
    this.blockstackSession = new blockstack.UserSession({appConfig: this.appConfig})
    var pending = false
    var signedIn = false

    if(this.blockstackSession.isUserSignedIn()) {
      signedIn = true
    }
    else if(!this.blockstackSession.isUserSignedIn() && this.blockstackSession.isSignInPending()) {
      signedIn = false
      pending = true
      this.blockstackSession.handlePendingSignIn().then(userData => {
        console.log("✓ Logged in as " + userData.username)
        this.setState({signedIn: true, pending: false})
       })
     }

     this.state = {
       pending,
       signedIn,
     }
  }

  signOut() {
    this.blockstackSession.signUserOut()
    this.setState({signedIn: false, pending: true})
    setInterval(() => this.setState({pending: false}), 3000) //there's no signUserOut() promise, so wait for sign out to happen.
  }

  render() {
    var signedIn = this.state.signedIn
    var pending = this.state.pending
    console.log(this.state)

    var displayItem
    if (pending) {
      displayItem = <LoadingPage />
    }
    else if (signedIn) {
      displayItem = <Editor blockstackSession={this.blockstackSession} signOut={() => this.signOut()}/>
    }
    else {
      displayItem = <LandingPage signIn={() => this.blockstackSession.redirectToSignIn()}/>
    }

    return (
      displayItem
    )
  }

}

class Editor extends React.Component {

  welcomeText = "To get started, just type!\n\nYou can bold and italicize with cmd-B and cmd-I.\n\nOnce the word vectors have initialized, word suggestions will appear as you type.\n\nProgress is saved instantly to the app - your text will be here when you come back."

  styles = {
    editor: {
      padding: "25px",
      width: "100%",
      height: "100%",
      fontSize: 18,
      fontFamily: "Sans-Serif",
      boxSizing: "border-box",
    }
  }

  //"absolute" is positioned relative to closet parent with "relative", default position is static
  createSuggestionsDecorator(blockKey, start, end, similarWords) {
    let AddSpan = (props) => {
      return (<span id="activeWord" style={{position: "relative"}}>
        {props.children}
      </span>)
    };
    let findWords = (contentBlock, callBack, contentState) => {
      if (contentBlock.getKey() == blockKey) {
        callBack(start, end)
      }
    }
    let decorator = new Draft.CompositeDecorator([
      {
        strategy: findWords,
        component: AddSpan,
      },
    ]);
    return decorator
  }

  constructor(props) {
    super(props)

    let onProgress = loadProgress => {this.setState({loadProgress})}

    // tf.setBackend("cpu") //using webgl backend causes slight freeze after every big matrix op
    // Disable vectors for now

    tf.setBackend('wasm')
      .then(() => fetchWordVectors(onProgress))
      .then(buffer => {
        this.setState({loadState: ["Loading word vectors", 1], loadProgress: 0})
        return loadWordVectors(buffer, onProgress)
      })
      .then((wordData) => {
        this.setState({
          wordVectors: wordData[0],
          wordDict: wordData[1],
          vectorsLoaded: true,
          loadState: ["Ready", 3]
        })
      })

    if (!this.props.blockstackSession.isUserSignedIn()) {
      console.log("ERROR: User not signed in, but should be")
    }

    const options = {decrypt: true}
    let savedEditor = this.props.blockstackSession.getFile("savedEditor", options).then(savedEditor => {
      if (savedEditor == null) {
        console.log("✓ Creating empty Draft editor")
        var newEditor = Draft.EditorState.createWithContent(Draft.ContentState.createFromText(this.welcomeText))
      } else {
        console.log("✓ Found saved Draft editor")
        var newEditor = Draft.EditorState.createWithContent(Draft.convertFromRaw(JSON.parse(savedEditor)))
      }
      this.setState({editorState: newEditor, finishedDownloadingEditorState: true})
    })

    this.state = {
      editorState: null,
      finishedDownloadingEditorState: false,
      activeWord: {text: "", blockKey: "", pos: null},
      similarWords: [],
      vectorsLoaded: false,
      wordDict: null,
      wordVectors: null,
      userData: null,
      loadProgress: 0.0,
      loadState: ["Downloading word vectors", 0], //disabling word vectors for now
      // loadState: ["Ready", 2],
      lastChange: Date.now(),
      lastChangeTimeout: null,

			textUnits: [],
			isWaitingForResults: false,
			searchText: "",
			errorMessage: "",
			examplePlaceholder: this.examplePlaceholder(),
      previousSentence: "",
      sentenceResultsCache: {},
    }

    let save = async () => {
      this.state.vectorsLoaded && this.setState({loadState: ["Saving", 2]})
      let rawEditorState = JSON.stringify(Draft.convertToRaw(this.state.editorState.getCurrentContent()))
      const options = {encrypt: true}
      await this.props.blockstackSession.putFile("savedEditor", rawEditorState, options)
      this.state.vectorsLoaded && this.setState({loadState: ["Saved", 2]})
    }
    setInterval(() => this.state.finishedDownloadingEditorState && save(), 10 * 1000)
  }

  handleKeyCommand(command, editorState) {
    const newState = Draft.RichUtils.handleKeyCommand(editorState, command);
    if (newState) {
      this.onChange(newState);
      return 'handled';
    }
    return 'not-handled';
  }

  onChange(editorState) {
    //Todo: editorState update only handled by second setState
    if (editorState.getLastChangeType() == "apply-entity") return //prevent shenanigans when editing happens in decorator

    //Below is book engine search updating logic
    this.setState((currentState, props) => {
      let activeSentence = this.getActiveSentence(editorState)
      let wasSelectionChange = currentState.editorState.getCurrentContent() == editorState.getCurrentContent() && currentState.editorState.getSelection() != editorState.getSelection() && editorState.getSelection().getHasFocus()
      let wasNewSentence = activeSentence.length == 1
      let wasNewParagraph = editorState.getLastChangeType() == "split-block" && !wasSelectionChange

      var newState = {}

      if (wasSelectionChange) {
        this.search(activeSentence)
        newState = {...newState, searchText: activeSentence}
      }
      else if (wasNewSentence || wasNewParagraph) {
        this.search(this.state.previousSentence)
        newState = {...newState, searchText: this.state.previousSentence}
      }

      return {...newState, previousSentence: activeSentence}

    })

    //Below is word-vector suggestion updating logic
    this.setState((currentState, props) => {
      let word = this.getActiveWord(editorState)
      let wordText = word.text && word.text.replace(",", "")
      wordText = wordText && wordText.replace(".", "")
      let lastChangeType = editorState.getLastChangeType()

      //only display suggestions after a 150ms pause.
      let lastChangeTimeDiff = Date.now() - this.state.lastChange
      const delayBeforeDisplay = 150//150
      let wasLongDelay = lastChangeTimeDiff > delayBeforeDisplay - 5// || lastChangeType != 'backspace-character'
      let lastChangeTimeout
      if (!wasLongDelay) {
        let updateEditor = () => {this.onChange(this.state.editorState)}
        lastChangeTimeout = setTimeout(updateEditor, delayBeforeDisplay)
      }
      clearTimeout(this.state.lastChangeTimeout)

      let wasCharacterEdit = lastChangeType == 'insert-characters' || lastChangeType == 'backspace-character'
      let wasSelectionChange = currentState.editorState.getCurrentContent() == editorState.getCurrentContent()
      let isNewWord = wordText != currentState.activeWord.text || word.blockKey != currentState.activeWord.blockKey
        || word.wordStart !=currentState.activeWord.wordStart
      let vectorAvailible = this.state.vectorsLoaded && wordText in this.state.wordDict.fromWord

      if((wasCharacterEdit || wasSelectionChange) && isNewWord && vectorAvailible && wasLongDelay) {
        this.getWordSuggestions(wordText).then(similarWords => {
          let wordDecorator = this.createSuggestionsDecorator(word.blockKey, word.wordStart, word.wordEnd, similarWords) //doing the update async didn't help freezing up
          this.setState((currentState, props) => {
            return {
              editorState: Draft.EditorState.set(currentState.editorState, {decorator: wordDecorator}),
              similarWords,
            }
          })
        })
        return {
          editorState: Draft.EditorState.set(editorState, {decorator: null}),
          activeWord: word,
          // similarWords,
          lastChange: Date.now(),
          lastChangeTimeout
        }
      }
      else if (!isNewWord) {
        return {editorState}
      }
      else {
        return {
          editorState: Draft.EditorState.set(editorState, {decorator: null}),
          activeWord: "",
          // similarWords: [],
          lastChange: Date.now(),
          lastChangeTimeout
        }
      }
    })
  }

  getActiveWord(editorState) {
      let selectionState = editorState.getSelection()
      let blockKey = selectionState.getAnchorKey()
      let currentContent = editorState.getCurrentContent();
      let currentContentBlock = currentContent.getBlockForKey(blockKey);
      let text = currentContentBlock.getText()
      let start = selectionState.getStartOffset()
      let end = selectionState.getEndOffset()
      let wordStart = text.lastIndexOf(" ", start-2) + 1
      let wordEnd = text.indexOf(" ", start-1)
      let word
      if (start != end) {
        word = null
      }
      else {
        if (wordStart == -1) wordStart = 0
        if (wordEnd == -1) wordEnd = text.length
        word = text.substring(wordStart, wordEnd).toLowerCase()
      }
      return {text: word, blockKey, wordStart, wordEnd}
  }

  getActiveSentence(editorState) {
      let selectionState = editorState.getSelection()
      let blockKey = selectionState.getAnchorKey()
      let currentContent = editorState.getCurrentContent();
      let currentContentBlock = currentContent.getBlockForKey(blockKey);
      let text = currentContentBlock.getText()
      let start = selectionState.getStartOffset()
      let wordStart = text.lastIndexOf(" ", start-2) + 1

      const cursorCharacter = "$$$^$$$" //since compromise.js (nlp(...) obj) does not support char pos
      text = text.slice(0, wordStart) + cursorCharacter + " " + text.slice(wordStart, text.length)
      let nlpResult = nlp(text)
      let sentences = nlpResult.sentences().list
      for (let sent of sentences) {
        if (cursorCharacter in sent.cache.words) {
          return sent.text().replace(cursorCharacter + " ", "")
        }
      }
      return null
  }

	search(searchText) {
		this.setState({isWaitingForResults: true})
		this.setState({examplePlaceholder: this.examplePlaceholder()})
    if (searchText in this.state.sentenceResultsCache) {
  			this.setState({
          textUnits: this.state.sentenceResultsCache[searchText],
          error: "",
          isWaitingForResults: false,
        })
    }
    else {
  		fetchFromServer(API_URL, {"sentence": searchText}).then(res => {
  			this.setState({
          textUnits: res.textUnits,
          error: "",
          isWaitingForResults: false,
          sentenceResultsCache: {...this.state.sentenceResultsCache, [searchText]: res.textUnits}
        })
  		}).catch(err => {
  			var errorMessage = (err.response && err.response.data.msg) || "Error calling API"
  			this.setState({isWaitingForResults: false, errorMessage})
  		})
    }
	}

  updateSearchText = e => this.setState({searchText: e.target.value})
  examplePlaceholder() {
    var randIdx = Math.floor(Math.random() * exampleQueries.length)
    var examplePlaceholder = "e.g. " + exampleQueries[randIdx]
    return examplePlaceholder
  }

  async getWordSuggestions(word) {
    //make word vector
    let wordIndex = this.state.wordDict.fromWord[word]
    let wordVectorsBuffer = await this.state.wordVectors.buffer()
    let vectorLength = this.state.wordVectors.shape[1]
    let vector = tf.buffer([vectorLength, 1], "float32")
    for (var idx = 0; idx < this.state.wordVectors.shape[1]; idx++) {
      vector.set(wordVectorsBuffer.get(wordIndex, idx), idx, 0)
    }
    vector = vector.toTensor()
    //get similarities
    let similarities = tf.matMul(this.state.wordVectors, vector)
    let similarityData = await similarities.data()
    similarities.dispose()
    vector.dispose()
    let similarWordIndexes = getHighestKIndices(similarityData, 10)
    let similarWords = similarWordIndexes.map(x => this.state.wordDict.fromIdx[x])
    return similarWords
  }

  componentDidMount() {
  }

  componentDidUpdate() {
    let currentWord = document.getElementById("activeWord")
    let lastPos = this.state.activeWord.pos
    if (currentWord) {
      let pos = currentWord.getBoundingClientRect()
      pos.y += window.scrollY
      if (lastPos == null || (pos.x != lastPos.x && pos.y != lastPos.y)) {
        this.setState({activeWord: {...this.state.activeWord, pos}})
      }
    }
    else if (!currentWord && lastPos != null) {
        this.setState({activeWord: {...this.state.activeWord, pos: null}})
    }
  }

  render() {
    let doRenderDropdown = this.state.activeWord.pos != null
    if (doRenderDropdown) {
      let pos = this.state.activeWord.pos
      var x = pos.x + pos.width + 10
      var y = pos.y
      // var x = pos.x
      // var  y = pos.y + 23
    }
    return (
      <div style={{height: "100%", width: "100%", display: "flex", justifyContent: "center", flexDirection: "row"}}>
        <div style={{width: "100%", height:"100%", display:"flex", flexDirection:"column", maxWidth: "800px"}}>
          <StatusBar state={this.state.loadState} progress={this.state.loadProgress} loaded={this.state.vectorsLoaded} logout={() => this.props.signOut()} />
          <div style={this.styles.editor}>
            {this.state.finishedDownloadingEditorState && <Draft.Editor
              editorState={this.state.editorState}
              onChange={this.onChange.bind(this)}
              handleKeyCommand={this.handleKeyCommand.bind(this)}
              style={{}}
            />}
          </div>
          {doRenderDropdown && <Dropdown similarWords={this.state.similarWords} position={{x, y}}/>}
        </div>
        <div style={{width: "100%", maxWidth:"350px", overflow: "auto"}}>
          <BookEngineListings
            onSearchSubmit={() => this.search(this.state.searchText)}
            searchText={this.state.searchText}
            onSearchChange={e => this.updateSearchText(e)}
            isWaitingForResults={this.state.isWaitingForResults}
            textUnits={this.state.textUnits}
            />
        </div>
      </div>
    )
  }
}

class Dropdown extends React.Component {

  style = {
    dropdown: {
      fontFamily: "Sans-Serif",
      borderRadius: "7px",
      boxShadow: "0px 1px 4px #ccc",
      overflow: "hidden",
      position: "absolute",
      backgroundColor: "white",
      zIndex: 100,
      MozUserSelect:"none",
      WebkitUserSelect:"none",
    },
    list: {
      listStyle: "none",
      padding: 0,
      margin: 0,
      fontSize: 18,//15
      boxSizing: "border-box",
    },
    item: {
      marginBottom: "-1px",
      borderBottom: "1px solid #ccc",
      padding: "2px",
      paddingLeft: "4px",
      paddingRight: "4px",
    }
  }

  constructor(props) {
    super(props)
  }

  render() {
    return (
      <div style={{top: this.props.position.y, left: this.props.position.x, ...this.style.dropdown}}>
        <ul style={this.style.list}>
        {this.props.similarWords.map((word) =>
          <div key={word}>
            <li style={this.style.item}>{word}</li>
          </div>
        )}
        </ul>
      </div>
    )
  }
}

class StatusBar extends React.Component {

  style = {
    bar: {
      backgroundColor: "#222",
      fontFamily: "Sans-Serif",
      height: "50px",
      width: "100%",
      color: "white",
      display: "flex",
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "center",
      paddingLeft: "10px",
      paddingRight: "12px",
      boxSizing: "border-box",
      fontSize: "16px",//"16px",
      boxShadow: "0px 1px 3px #ccc",

    }
  }

  buttonStyle = {
    color: "white",
    backgroundColor: "rgba(0, 0, 0, 0)",
    borderRadius: "2px 2px 2px 2px",
    fontSize: "16px",
    borderColor: "#000",
    padding: "4 6",
    cursor: "pointer",
    borderWidth: "1px 1px 1px 1px",
    borderStyle: "solid",
    borderColor: "white"
    // width: "67px",
  }

  statusStyle = {
    color: "white",
    backgroundColor: "rgba(0, 0, 0, 0)",
    fontSize: "16px",
    borderRadius: "2px 2px 2px 2px",
    borderColor: "#000",
    padding: "4 6",
    borderWidth: "0px",
    // borderWidth: "1px 1px 1px 1px",
    borderStyle: "solid",
    borderColor: "white"
  }

  constructor(props) {
    super(props)
  }
  render () {
    let doShowProgress = this.props.state[1] == 0 || this.props.state[1] == 1
    let progress = "(" + Math.round(this.props.progress * 100).toString() + "%)"
    return (
      <div style={this.style.bar}>
        <div><button style={this.statusStyle}>{this.props.state[0] + " "} {doShowProgress && progress}</button></div>
        <div><button style={this.buttonStyle} onClick={() => this.props.logout()}>Logout</button></div>
      </div>
    )
  }
}

class BookEngineListings extends React.Component {

	style = {
    width: "100%",
		fontSize: "16px",
		fontFamily: "Arial",
		lineHeight: 1.3
	}

	headerStyle = {
		fontSize: "35px",
    height: "50px",
    paddingTop: "5px",
		paddingLeft: "10px",
		paddingRight: "10px",
    boxSizing: "border-box",
	}

	constructor(props) {
		super(props)
		this.state = {}
	}

	render() {
		return (
		 <div style={{height: "100%", display: "flex", justifyContent: "center"}}>
			 <div style={this.style}>
				 <div style={this.headerStyle}>Book Engine</div>
         <div style={{paddingLeft: "10px", paddingRight: "10px"}}>
  				 <div style={{marginBottom: "20px", marginTop: "15px"}}>
  					 <SearchBar
  						 onSubmit={this.props.onSearchSubmit}
  						 onChange={this.props.onSearchChange}
  						 disabled={this.props.isWaitingForResults}
  						 value={this.props.searchText}
  						 isLoading={this.props.isWaitingForResults}
  						/>
  				 </div>
  				{this.props.textUnits.map(unit => <ResultItem key={unit.vectorNum} unit={unit} />)}
  				<div style={{marginTop: "30px", marginBottom: "30px", border: "1px solid #fff", borderRadius: "4px", padding: "5px"}}>
  				</div>
        </div>
			 </div>
		 </div>
		)
	}

}

class ResultItem extends React.Component {

	style = {
		// boxShadow: "0px 0px 3px #ccc",
		padding: "20px",
		paddingTop: "20px",
		paddingBottom: "20px",
		borderRadius: "4px",
	}

	textAreaStyle = {
		cursor: "pointer",
		padding: "5px",
		paddingRight: "15px", //scroll bar space
	}

	infoStringStyle = {
		marginTop: "20px",
		textAlign: "right",
		fontSize: "16px",
		width: "60%",
		padding: "5px",
		borderRadius: "5px",
		border: "1px solid #000",
	}

	constructor(props) {
		super(props)
		this.state = {
			expanded: false,
			hovering: false,
			isLoading: false,
			units: [this.props.unit],
		}
		this.textAreaRef = React.createRef()
	}

	formatText(text) {
		text = text.replace("--", "—") //em dash
		var parts = text.split(/(_.+?_|=.+?=)/g)
		for(var i=1; i<parts.length; i+=2) {
			if (parts[i][0] == "_") {
				var part = parts[i].slice(1, parts[1].length-1)
				parts[i] = <b key={i}>{part}</b>
			}
			else if (parts[i][0] == "=") {
				var part = parts[i].slice(1, parts[1].length-1)
				parts[i] = <i key={i}>{part}</i>
			}
		}
		return parts
	}

	range(start, end) {
		var range = []
		for (var i=start; i<end; i++) {
			range.push(i)
		}
		return range
	}

	fetchMoreText() {
		var numToAdd = 15
		var bookNum = this.props.unit.bookNum
		if (this.state.isLoading) {
			return
		}
		this.setState({isLoading: true})
		var start = this.state.units[this.state.units.length-1].inBookLocation + 1
		var inBookLocations = this.range(start, start+numToAdd)
		var bottomPromise = fetchFromServer(API_HELPER_URL, {bookNum, inBookLocations})
		var end = this.state.units[0].inBookLocation
		var inBookLocations= this.range(end-numToAdd, end)
		var topPromise = fetchFromServer(API_HELPER_URL, {bookNum, inBookLocations})
		//Updates bottom first, then top
		Promise.all([bottomPromise, topPromise]).then(allRes => {
			this.setState(state => ({units: [...state.units, ...allRes[0].textUnits]}), () => {
				this.setState(state => {
					var prevHeight = this.textAreaRef.current.scrollHeight
					var prevScrollTop = this.textAreaRef.current.scrollTop //chrome modifies scrollTop when content added to top
					return ({units: [...allRes[1].textUnits, ...state.units], isLoading: false, prevHeight, prevScrollTop})
				}, () => {
					var heightDiff = this.textAreaRef.current.scrollHeight - this.state.prevHeight
					this.textAreaRef.current.scrollTop = this.state.prevScrollTop + heightDiff
				})
			})
		})
	}

	render() {
		var unit = this.props.unit
		var onHover = e => this.setState({hovering: true})
		var onUnHover = e => this.setState({hovering: false})
		var onClick = e => {
			this.setState({expanded: true})
			this.fetchMoreText()
		}
		// var onClick = e => this.setState(state => ({expanded: !state.expanded}))
		var hoverStyle = this.state.hovering ? {boxShadow: "0px 0px 3px #ccc"} : {}
		var expandedStyle = this.state.expanded ? {height: "80%", overflow: "auto"} : {}
		var fields = ["Title", "Author", "Author Birth", "Author Death"]
		var [title, author, birth, death] = fields.map(k => (unit[k][0] ? unit[k][0] : "?"))
		var includeAuthorString = (birth != "?") || (death != "?")
		var authorString = "(" + birth + " - " + death + ")"
		var infoString = title + " — " + author + " " + (includeAuthorString ? authorString : "")
		var infoDiv = <div style={this.infoStringStyle}>{infoString}</div>

		//can make heuristic: table of contents (e.g. search "test12345") have a bunch of em dashes in them.
		// So say if the number of dashes exceeds ten, then break at those dashes.

		return (
			<React.Fragment>
				<div style={{...this.style, ...hoverStyle}} onMouseOver={onHover} onMouseLeave={onUnHover}>
					<div onClick={onClick} ref={this.textAreaRef} style={{...this.textAreaStyle, ...expandedStyle}}>
						<div>{this.formatText(this.state.units[0]["textUnit"])}</div>
						{this.state.units.slice(1).map(tu => (
							<React.Fragment key={tu["inBookLocation"]}>
								<br></br>
								<div>{this.formatText(tu["textUnit"])}</div>
							</React.Fragment>
						))}
					</div>
					<div style={{display: "flex", justifyContent: "flex-end"}}>
						{this.state.expanded && infoDiv}
					</div>
				</div>
				<div style={{width: "100%", border: "none", height: "1px", backgroundColor: "black"}}/>
			</React.Fragment>
		)
	}
}

class SearchBar extends React.Component{

	style = {
		borderRadius: "4px",
		width: "100%",
		boxShadow: "0px 1px 4px #ccc",
		boxSizing: "border-box",
		display: "flex",
	}

	inputStyle = {
		borderRadius: "4px",
		border: "none",
		outline: "none",
		padding: "10px",
    overflow: "auto",
		fontFamily: "inherit",
		fontSize: "inherit",
		width: "100%",
		backgroundColor: "white",
	}

  constructor(props){
    super(props);
  }

  render() {
    let onSubmit = e => {e.preventDefault(); this.props.onSubmit()}
    return (
      <div style={this.style}>
        <form onSubmit={onSubmit} style={{margin: 0, width: "100%"}}>
          <input style={this.inputStyle}
						placeholder={this.props.placeholder}
						onChange={this.props.onChange}
						disabled={this.props.disabled}
						value={this.props.value}
					/>
        </form>
				<div style={{display: "flex", justifyContent: "center", alignItems: "center", paddingRight: "5px"}}>
					<LoadingIndicator isLoading={this.props.isLoading} onClicked={this.props.onSubmit} />
				</div>
      </div>
    )
  }
}

class LandingPage extends React.Component {

  exampleStyle = {
    sentence: {
      fontSize: "24px",
      fontFamily: "sans-serif",
    },
    dropdown: {
      fontFamily: "Sans-Serif",
      borderRadius: "7px",
      boxShadow: "0px 1px 4px #ccc",
      // overflow: "hidden",
      backgroundColor: "white",
      zIndex: 100,
      marginTop: "1.9em",
      position: "relative",
      left: "-3.8em",
      marginRight: "-3.8em",
      // MozUserSelect:"none",
      // WebkitUserSelect:"none",
    },
    list: {
      listStyle: "none",
      padding: 0,
      margin: 0,
      fontSize: 20,
      boxSizing: "border-box",
    },
    item: {
      marginBottom: "-1px",
      borderBottom: "1px solid #ccc",
      padding: "4px",
      paddingLeft: "6px",
      paddingRight: "6px",
    }
  }

  headerStyle = {
    fontFamily: "LyonDisplay,Georgia,serif",
    fontSize: "60px",
    marginBottom: "0px",
    marginTop: "20px",
    // letterSpacing: "0.01em",
  }

  buttonStyle = {
    fontFamily: "LyonDisplay,Georgia,serif",
    color: "black",
    fontSize: "24px",
    backgroundColor: "white",
    borderRadius: "2px 2px 2px 2px",
    borderColor: "#000",
    padding: "15 30",
    cursor: "pointer",
    borderWidth: "1px 1px 1px 1px",
    borderStyle: "solid",
    // width: "67px",
  }

  constructor(props) {
    super(props)
    this.state = {
      blink: true,
    }
    setTimeout(() => this.toggleBlink(), 500)
  }

  toggleBlink() {
    this.setState(oldState => {return {blink: !oldState.blink}})
    setTimeout(() => this.toggleBlink(), 500)
  }

  render() {
    return (
      <div style={{display: "flex", flexDirection: "column", alignItems:"center", boxSizing: "border-box", margin: "20px"}}>
        <h1 style={this.headerStyle}>Prosepaper</h1>

        <div style={{display: "flex", flexDirection: "row", marginTop: "130px", marginBottom: "120px"}}>

          <div style={this.exampleStyle.sentence}>A word processor that actually knows a bit about words <span style={{color: this.state.blink ? "black" : "white"}}>|</span>
          <br></br><br></br><br></br><br></br><br></br><br></br>


          </div>

          <div style={this.exampleStyle.dropdown}>
            <ul style={this.exampleStyle.list}>
              <div><li style={this.exampleStyle.item}>phrases</li></div>
              <div><li style={this.exampleStyle.item}>uttered</li></div>
              <div><li style={this.exampleStyle.item}>language</li></div>
              <div><li style={this.exampleStyle.item}>expression</li></div>
              <div><li style={this.exampleStyle.item}>letters</li></div>
              <div><li style={this.exampleStyle.item}>write</li></div>
              <div><li style={this.exampleStyle.item}>text</li></div>
              <div><li style={this.exampleStyle.item}>references</li></div>
              <div><li style={this.exampleStyle.item}>describes</li></div>
              <div><li style={this.exampleStyle.item}>meaning</li></div>
            </ul>
          </div>

        </div>

        <div style={{...this.exampleStyle.sentence, fontFamily: "LyonDisplay,Georgia,serif", marginBottom: "60px"}}><b>Get realtime word suggestions while typing</b></div>
        <button style={this.buttonStyle} onClick={() => this.props.signIn()}>Sign in with Blockstack
        </button>
      </div>
    )
  }
}

class LoadingPage extends React.Component {
  constructor(props) {
    super(props)
  }

  render() {
    return (
      <div style={{display: "flex", justifyContent:"center", alignItems:"center", width:"100%", height:"100%"}}>
        <LoadingIndicator isLoading={true}/>
      </div>
    )
  }

}

class LoadingIndicator extends React.Component {

	containerStyle = {
		display: "flex",
		justifyContent: "center",
		alignItems: "center",
		backgroundColor:"white",
		borderRadius:"50%",
		height:"30px",
		width:"30px",
		cursor: "pointer",
	}

  constructor(props) {
    super(props)
    this.iconRef = React.createRef()
  }

  updateAnimation() {
    if (this.props.isLoading) {
      this.iconRef.current.style.animationPlayState = "running"
    }
    else {
      this.iconRef.current.style.animationPlayState = "paused"
    }
  }

  componentDidMount() {
    this.iconRef.current.style.animation = "rotating 0.5s linear infinite"
    this.updateAnimation()
  }
  componentDidUpdate() {
    this.updateAnimation()
  }

  render() {
    var active = true
    return (
      <div ref={this.iconRef} style={this.containerStyle}>
        <div author="Loading icon by aurer: https://codepen.io/aurer/pen/jEGbA" title="0">
          <svg version="1.1" id="loader-1" x="0px" y="0px"
           width="30px" height="30px" viewBox="0 0 40 40" enableBackground="new 0 0 40 40" space="preserve">
            <path opacity="0.2" fill="#000" d="M20.201,5.169c-8.254,0-14.946,6.692-14.946,14.946c0,8.255,6.692,14.946,14.946,14.946
              s14.946-6.691,14.946-14.946C35.146,11.861,28.455,5.169,20.201,5.169z M20.201,31.749c-6.425,0-11.634-5.208-11.634-11.634
              c0-6.425,5.209-11.634,11.634-11.634c6.425,0,11.633,5.209,11.633,11.634C31.834,26.541,26.626,31.749,20.201,31.749z"/>
            <path fill="#000" d="M26.013,10.047l1.654-2.866c-2.198-1.272-4.743-2.012-7.466-2.012h0v3.312h0
              C22.32,8.481,24.301,9.057,26.013,10.047z">
            </path>
          </svg>
        </div>
      </div>
    )
  }

}

ReactDOM.render(<App />, document.getElementById("root"))
